"""Manager GRPO training loop.

Wires together:
  - SubagentPool (three frozen tools) OR RemoteSubagentPool (vLLM HTTP server)
  - ManagerToolEnvironment OR argument-binding tools
  - GRPOTrainer with the binary correctness reward
  - W&B logging (optional)

For full-parameter GRPO on 8B+ models with 4 GPUs, set subagent_server_url in
ManagerGRPOConfig to point at a vLLM server (GPU 0) serving the three subagents.
The training processes (GPUs 1-3) then carry zero subagent weight, leaving all
VRAM for ZeRO Stage 3 manager sharding. See scripts/start_subagent_server.sh
and scripts/train_manager_grpo_multigpu.sh for the launch workflow.
"""
from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except Exception:
    TRL_AVAILABLE = False

try:
    from trl.chat_template_utils import add_response_schema
    HAS_RESP_SCHEMA = True
except Exception:
    HAS_RESP_SCHEMA = False

from ..benchmarks.base import StandardRow
from ..subagents.runtime import (
    FrozenSubagent,
    RemoteSubagentPool,
    SubagentPool,
    default_subagent_max_new_tokens,
)
from ..utils.chat_template import (
    ensure_nothink_chat_template,
    generation_prompt_opens_think,
)
from ..utils.io import write_json
from ..utils.modeling import load_text_causal_lm
from ..utils.seed import set_seed
from .prompt import build_manager_system_prompt, build_manager_user_message
from .reward import build_reward_funcs


def _grpo_supports_environment_factory() -> bool:
    if not TRL_AVAILABLE:
        return False
    try:
        sig = inspect.signature(GRPOTrainer.__init__)
        return "environment_factory" in sig.parameters
    except Exception:
        return False


# ----------------------- Tool environment & registry -----------------------

# Module-level state used by the tool functions and environment. We keep this
# minimal and reset it at the start of every train_manager_grpo() call.
_POOL: Optional[SubagentPool] = None
_ROW_INDEX: Dict[int, StandardRow] = {}


def _ensure_pool() -> SubagentPool:
    if _POOL is None:
        raise RuntimeError("SubagentPool not initialized. Call _init_globals first.")
    return _POOL


def _init_globals(pool: SubagentPool, rows: List[StandardRow]) -> None:
    global _POOL, _ROW_INDEX
    _POOL = pool
    _ROW_INDEX = {int(r.example_id): r for r in rows}


def _run_tool(
    agent_kind: str,
    example_id: int,
    namespace: str = "default",
    candidate_answer: str = "",
) -> str:
    eid = int(example_id)
    row = _ROW_INDEX.get(eid)
    if row is None:
        return '{"error": "example_id not found"}'
    pool = _ensure_pool()
    return pool.call(
        agent_kind=agent_kind,
        example_id=eid,
        question=row.question,
        context=row.context,
        choices=row.choices,
        cache_namespace=namespace,
        candidate_answer=candidate_answer,
    )


# Argument-binding tool functions (used when binding_mode == "argument")
def extractor_tool(example_id: int) -> str:
    """Extract decision-relevant signals for the given example.

    Args:
        example_id: The current example's ID from the user message.

    Returns:
        JSON string with extracted facts and key evidence.
    """
    return _run_tool("extractor", int(example_id))


def reasoner_tool(example_id: int) -> str:
    """Produce a structured reasoning scaffold for the given example.

    Args:
        example_id: The current example's ID from the user message.

    Returns:
        JSON string with sub-questions, knowledge, candidate analysis.
    """
    return _run_tool("reasoner", int(example_id))


def verifier_tool(example_id: int, current_draft: str = "") -> str:
    """Audit reasoning for domain principles and potential errors.

    Args:
        example_id: The current example's ID from the user message.
        current_draft: Your current draft answer key (e.g. "B") to audit.

    Returns:
        JSON string with relevant principles, checks, and potential errors.
    """
    return _run_tool("verifier", int(example_id), candidate_answer=str(current_draft or ""))


class ManagerToolEnvironment:
    """Environment-binding alternative: tools don't take an example_id arg."""

    def reset(self, example_id: int, **kwargs) -> Optional[str]:
        """Bind this rollout to the given example.

        Args:
            example_id: ID of the sampled training example.

        Returns:
            None.
        """
        self.example_id = int(example_id)
        self._called = set()
        return None

    def _guard_repeat(self, kind: str) -> Optional[str]:
        called = getattr(self, "_called", None)
        if called is None:
            called = set()
            self._called = called
        if kind in called:
            return '{"error": "tool_already_called", "detail": "each tool may be used at most once"}'
        called.add(kind)
        return None

    def extractor_tool(self) -> str:
        """Extract decision-relevant signals for the current example.

        Returns:
            JSON string with extracted facts.
        """
        err = self._guard_repeat("extractor")
        if err:
            return err
        return _run_tool("extractor", getattr(self, "example_id", -1))

    def reasoner_tool(self) -> str:
        """Produce a reasoning scaffold for the current example.

        Returns:
            JSON string with reasoning structure.
        """
        err = self._guard_repeat("reasoner")
        if err:
            return err
        return _run_tool("reasoner", getattr(self, "example_id", -1))

    def verifier_tool(self, current_draft: str = "") -> str:
        """Audit reasoning for domain principles and potential errors.

        Args:
            current_draft: Your current draft answer key (e.g. "B") to audit.

        Returns:
            JSON string with relevant principles, checks, and potential errors.
        """
        err = self._guard_repeat("verifier")
        if err:
            return err
        return _run_tool(
            "verifier",
            getattr(self, "example_id", -1),
            candidate_answer=str(current_draft or ""),
        )


# ----------------------- Tool interface pre-flight -----------------------

_PREFLIGHT_TOOL_CALL = {
    "id": "preflight_1",
    "type": "function",
    "function": {"name": "verifier_tool", "arguments": {"current_draft": "B"}},
}


def preflight_tool_interface_check(tok, chat_template_kwargs: Dict[str, Any]) -> None:
    """Fail fast if the template <-> parser round trip cannot carry a tool call.

    The failure mode this guards against is *silent*: if the generation prompt
    ends with an open ``<think>`` tag, or the response template/schema does not
    match the format the chat template renders, TRL decodes every completion
    without a ``tool_calls`` field, no tool ever executes, tools/call_frequency
    logs 0.0 forever, and the reward collapses — with no error raised anywhere.
    Set AGENT_ROUTING_SKIP_PREFLIGHT=1 to bypass (not recommended).
    """
    if os.environ.get("AGENT_ROUTING_SKIP_PREFLIGHT", "0") == "1":
        print("[MANAGER_GRPO] preflight skipped via AGENT_ROUTING_SKIP_PREFLIGHT=1")
        return

    # 1) In non-thinking mode, the generation prompt must not end with an open
    #    <think> tag — with the configured kwargs AND without them (the bare
    #    render covers TRL/vLLM render sites that may not forward
    #    chat_template_kwargs). In thinking mode an open <think> is expected:
    #    the model closes it before calling tools, which parses fine.
    thinking_mode = bool(chat_template_kwargs.get("enable_thinking"))
    if not thinking_mode and generation_prompt_opens_think(tok, chat_template_kwargs):
        raise RuntimeError(
            "Tool-call preflight failed: the generation prompt ends with an "
            "open <think> tag even with the configured chat_template_kwargs "
            f"({chat_template_kwargs}). The tool-call parser will swallow "
            "every completion into reasoning_content and the tool call rate "
            "will be 0. Use a nothink chat template (see "
            "src/utils/chat_template.ensure_nothink_chat_template) or set "
            "enable_thinking accordingly."
        )
    if not thinking_mode and generation_prompt_opens_think(tok):
        print(
            "[MANAGER_GRPO] WARNING: the chat template still opens <think> when "
            "enable_thinking is not passed. Render sites that drop "
            "chat_template_kwargs will silently break tool-call parsing."
        )

    # 2) Round-trip a native tool call through the template and the parser the
    #    trainer will actually use (response_template / response_schema).
    has_parser = (
        getattr(tok, "response_template", None) is not None
        or getattr(tok, "response_schema", None) is not None
    )
    if not has_parser:
        raise RuntimeError(
            "Tool-call preflight failed: the tokenizer has neither "
            "response_template nor response_schema, so TRL will decode "
            "completions as plain text and never see a tool call. "
            "trl.chat_template_utils.add_response_schema most likely failed "
            "to recognize this chat template — check the TRL version."
        )
    conversation = [
        {"role": "user", "content": "preflight"},
        {"role": "assistant", "content": "DRAFT_ANSWER_B", "tool_calls": [_PREFLIGHT_TOOL_CALL]},
    ]
    prompt_ids = tok.apply_chat_template(
        conversation[:1], add_generation_prompt=True, tokenize=True,
        **chat_template_kwargs,
    )
    full_text = tok.apply_chat_template(
        conversation, add_generation_prompt=False, tokenize=False,
        **chat_template_kwargs,
    )
    prompt_text = tok.apply_chat_template(
        conversation[:1], add_generation_prompt=True, tokenize=False,
        **chat_template_kwargs,
    )
    if not full_text.startswith(prompt_text):
        raise RuntimeError(
            "Tool-call preflight failed: the chat template is not "
            "prefix-preserving for a tool-calling assistant turn."
        )
    completion_ids = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    try:
        from trl.chat_template_utils import parse_response as _trl_parse_response
        parsed = _trl_parse_response(tok, completion_ids, prefix=prompt_ids)
    except ImportError:
        parsed = tok.parse_response(completion_ids, prefix=prompt_ids)
    if not parsed.get("tool_calls"):
        raise RuntimeError(
            "Tool-call preflight failed: a tool call rendered by this chat "
            "template does not parse back into `tool_calls` "
            f"(parsed={parsed!r}). The trainer would log tools/call_frequency "
            "= 0 forever. Check that the response template/schema matches the "
            "chat template (trl.chat_template_utils.add_response_schema) and "
            "that the template does not default to thinking mode."
        )
    print(
        "[MANAGER_GRPO] tool-interface preflight OK: "
        f"parsed tool_calls={[(tc.get('function') or {}).get('name') for tc in parsed['tool_calls']]}, "
        f"content={parsed.get('content')!r}"
    )


# ----------------------- Trainer entry point -----------------------

@dataclass
class ManagerGRPOConfig:
    base_model: str
    rows: List[StandardRow]              # filtered to a training split
    out_dir: str
    extractor_adapter: Optional[str]
    reasoner_adapter: Optional[str]
    verifier_adapter: Optional[str]
    manager_adapter: Optional[str] = None
    fail_buffer_jsonl: Optional[str] = None
    raw_trace_jsonl: Optional[str] = None
    seed: int = 42
    per_device_train_batch_size: int = 2
    max_completion_length: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    enable_thinking: bool = False
    num_generations: int = 6
    generation_batch_size: int = 0
    learning_rate: float = 1e-6
    grpo_beta: float = 0.001
    max_steps: int = -1
    routing_efficiency_bonus: float = 0.0
    tool_use_bonus: float = 0.0
    ccr_mode: bool = False               # if true, use log-scoring-rule CCR reward instead of binary
    ccr_p_high: float = 0.9             # CCR implicit confidence when 0 tools called
    ccr_p_low: float = 0.2              # CCR implicit confidence when ccr_k_max tools called
    ccr_k_max: int = 3                  # must match max_tool_calling_iterations
    full_parameter_rl: bool = False      # if true, merge init adapter and train all model weights
    binding_mode: str = "auto"           # auto | environment | argument
    subagent_server_url: Optional[str] = None  # if set, call subagents via vLLM HTTP (no local model load)
    use_wandb: bool = False
    wandb_project: str = "agent_routing"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"
    task_description: str = ""           # optional, passed into manager system prompt
    exploration_hint: str = ""           # START-style hint injected into system prompt during training only
    clip_epsilon_high: float = 0.0       # DAPO Clip-Higher: asymmetric clip upper bound (0 = symmetric/standard)
    # Adaptive Deliberation Control reward (replaces CCR)
    adc_mode: bool = False               # enable ADC anytime per-draft reward (recommended over ccr_mode)
    adc_cost_per_tool: float = 0.02      # per-tool cost TARGET; calibrate to <= 1/3-1/2 of the empirical
                                         # marginal tool value (corrections-corruptions)/tool_calls
    adc_draft_bonus: float = 0.02        # bonus scale for the anytime draft-correctness average. Kept small:
                                         # any draft-content bonus taxes corrected trajectories and subsidizes
                                         # k=0; honest drafts are already environment-incentivized via
                                         # verifier_tool(current_draft) -> final accuracy
    adc_missing_draft_penalty: float = 0.1  # penalty per tool call without an accompanying draft (format-only)
    adc_final_bonus: float = 1.0         # bonus for final correct answer
    adc_variant: str = "anytime"         # anytime | transition | sum (latter two: ablation arms only)
    adc_format_penalty: float = 0.2      # flat penalty on policy-chosen format violations (reward clamped to
                                         # min(r,0)-penalty; truncated rollouts are exempt)
    adc_cost_warmup_steps: int = 100     # linearly ramp cost_per_tool 0 -> target over N steps (0 = off);
                                         # lets tool skills form before parsimony pressure applies
    scale_rewards: str = "none"          # GRPO advantage scaling: none|batch|group. "group" amplifies the
                                         # tiny -cost*k gaps in all-correct groups into full-size negative
                                         # advantages and collapses tool use as accuracy rises. "none"
                                         # (Dr. GRPO) keeps cost at its nominal scale; "batch" is a middle
                                         # ground if parsimony learns too slowly under "none".


def train_manager_grpo(cfg: ManagerGRPOConfig) -> None:
    if not TRL_AVAILABLE:
        raise RuntimeError("trl is required for manager GRPO training.")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Resolve binding mode ----
    binding_mode = cfg.binding_mode
    if binding_mode == "auto":
        binding_mode = "environment" if _grpo_supports_environment_factory() else "argument"
    if binding_mode == "environment" and not _grpo_supports_environment_factory():
        raise RuntimeError(
            "binding_mode=environment requires a TRL version with environment_factory."
        )
    print(f"[MANAGER_GRPO] binding_mode={binding_mode}")

    # ---- Build subagent pool ----
    if cfg.subagent_server_url:
        # Multi-GPU mode: subagents are served by a vLLM process on a dedicated GPU.
        # No model weights are loaded here; all VRAM stays available for ZeRO3.
        pool = RemoteSubagentPool(
            server_url=cfg.subagent_server_url,
            registered_kinds=["extractor", "reasoner", "verifier"],
        )
        print(f"[MANAGER_GRPO] using remote subagent pool -> {cfg.subagent_server_url}")
    else:
        pool = SubagentPool()
        if cfg.extractor_adapter:
            pool.register(FrozenSubagent(cfg.base_model, cfg.extractor_adapter, "extractor", device))
        if cfg.reasoner_adapter:
            pool.register(FrozenSubagent(cfg.base_model, cfg.reasoner_adapter, "reasoner", device))
        if cfg.verifier_adapter:
            pool.register(FrozenSubagent(cfg.base_model, cfg.verifier_adapter, "verifier", device))
        if not pool._agents:
            raise ValueError(
                "At least one subagent adapter must be provided, "
                "or set subagent_server_url to use a vLLM server."
            )
        print(f"[MANAGER_GRPO] subagents loaded: {sorted(pool._agents.keys())}")

    _init_globals(pool, cfg.rows)

    # ---- Side-channel files ----
    fail_buffer = cfg.fail_buffer_jsonl or os.path.join(cfg.out_dir, "fail_buffer.jsonl")
    raw_trace = cfg.raw_trace_jsonl or os.path.join(cfg.out_dir, "train_raw_trace.jsonl")
    is_main = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0

    if is_main:
        os.makedirs(os.path.dirname(fail_buffer) or ".", exist_ok=True)
        if os.environ.get("FAIL_BUFFER_APPEND", "0") == "0":
            open(fail_buffer, "w", encoding="utf-8").close()
        os.makedirs(os.path.dirname(raw_trace) or ".", exist_ok=True)
        if os.environ.get("RAW_TRACE_APPEND", "0") == "0":
            open(raw_trace, "w", encoding="utf-8").close()
        print(f"[MANAGER_GRPO] fail_buffer -> {fail_buffer}")
        print(f"[MANAGER_GRPO] raw_trace   -> {raw_trace}")

    # ---- W&B ----
    if cfg.use_wandb:
        try:
            import wandb  # noqa: F401
        except Exception as e:
            raise RuntimeError("`wandb` not installed but use_wandb=True.") from e
        if cfg.wandb_project:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if cfg.wandb_entity:
            os.environ["WANDB_ENTITY"] = cfg.wandb_entity
        os.environ["WANDB_MODE"] = cfg.wandb_mode
        run_name = cfg.wandb_run_name or f"grpo_{os.path.basename(cfg.out_dir.rstrip('/'))}_{int(time.time())}"
        os.environ["WANDB_NAME"] = run_name
        print(f"[WANDB] {cfg.wandb_project} / {run_name}")
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    # ---- Build training dataset ----
    label_keys_per_row: List[List[str]] = [list(r.choices.keys()) for r in cfg.rows]

    def _to_record(r: StandardRow, keys: List[str]) -> Dict[str, Any]:
        sys_prompt = build_manager_system_prompt(
            label_keys=keys,
            task_description=cfg.task_description,
            exploration_hint=cfg.exploration_hint,
        )
        user_msg = build_manager_user_message(
            example_id=r.example_id,
            question=r.question,
            context=r.context,
            choices=r.choices,
            binding_mode=binding_mode,
        )
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            "ground_truth": r.ground_truth,
            "example_id": int(r.example_id),
            "choice_keys": list(r.choices.keys()),
        }

    train_records = [_to_record(r, k) for r, k in zip(cfg.rows, label_keys_per_row)]
    train_dataset = Dataset.from_list(train_records)

    # ---- Manager model + tokenizer ----
    # Resolve relative local paths to absolute so transformers doesn't mistake
    # them for HuggingFace repo IDs (e.g. "outputs/manager/..." → OSError).
    # Detection: a HF repo ID has exactly one "/" and no OS path separators;
    # a local path has multiple "/" or contains os.sep (backslash on Windows).
    if cfg.manager_adapter:
        p = cfg.manager_adapter
        looks_local = os.sep in p or p.count("/") > 1 or p.startswith(".")
        if looks_local:
            cfg.manager_adapter = os.path.abspath(p)

    manager_tok = AutoTokenizer.from_pretrained(
        cfg.manager_adapter or cfg.base_model, trust_remote_code=True
    )
    manager_tok.padding_side = "left"
    if manager_tok.pad_token_id is None and manager_tok.eos_token_id is not None:
        manager_tok.pad_token_id = manager_tok.eos_token_id

    # Qwen3.5 ships the THINK template variant: by default the generation
    # prompt ends with an open `<think>` tag, and the tool-call parser then
    # swallows the whole (non-thinking) completion into reasoning_content —
    # tool call rate reads exactly 0, silently. Normalize the template default
    # to nothink BEFORE the response schema is chosen, so both match.
    if not cfg.enable_thinking:
        ensure_nothink_chat_template(manager_tok, tag="MANAGER_GRPO")

    if HAS_RESP_SCHEMA:
        try:
            manager_tok = add_response_schema(manager_tok)
        except Exception as e:
            # Do NOT swallow this silently: without a response template/schema
            # TRL falls back to plain-text decoding and never parses a tool
            # call. The preflight below turns this into a hard error.
            print(f"[MANAGER_GRPO] WARNING: add_response_schema failed: {e}")

    preflight_tool_interface_check(
        manager_tok, {"enable_thinking": bool(cfg.enable_thinking)}
    )

    # Tool results are appended INSIDE the completion budget: TRL rolls back
    # any tool result that would push the sequence past max_completion_length,
    # so a budget smaller than a single subagent reply makes every tool call
    # useless even when parsing works.
    _max_tool_tokens = max(
        default_subagent_max_new_tokens(k)
        for k in ("extractor", "reasoner", "verifier")
    )
    _recommended = _max_tool_tokens * 2 + 512
    if cfg.max_completion_length < _max_tool_tokens + 256:
        print(
            f"[MANAGER_GRPO] WARNING: max_completion_length="
            f"{cfg.max_completion_length} cannot fit a single largest tool "
            f"reply ({_max_tool_tokens} tokens) plus the manager's turns. "
            f"TRL will roll the tool results back and trajectories will end "
            f"tool-less. Use at least {_recommended} "
            f"(--mgr_max_completion_length {_recommended}) or lower "
            f"SUBAGENT_*_MAX_NEW_TOKENS."
        )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    is_full_init = (
        cfg.manager_adapter
        and os.path.isdir(cfg.manager_adapter)
        and os.path.exists(os.path.join(cfg.manager_adapter, "config.json"))
        and not os.path.exists(os.path.join(cfg.manager_adapter, "adapter_config.json"))
    )
    if cfg.full_parameter_rl and is_full_init:
        manager_model = load_text_causal_lm(
            cfg.manager_adapter, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        print(f"[MANAGER_GRPO] full-parameter init model -> {cfg.manager_adapter}")
    else:
        manager_model = load_text_causal_lm(
            cfg.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        if cfg.manager_adapter:
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft is required to load --mgr_init_adapter.")
            manager_model = PeftModel.from_pretrained(
                manager_model,
                cfg.manager_adapter,
                is_trainable=(not cfg.full_parameter_rl),
            ).to(device)
            if cfg.full_parameter_rl:
                manager_model = manager_model.merge_and_unload().to(device)
                print(f"[MANAGER_GRPO] merged init adapter for full-parameter RL -> {cfg.manager_adapter}")
            else:
                print(f"[MANAGER_GRPO] manager init adapter -> {cfg.manager_adapter}")
    if cfg.full_parameter_rl:
        for p in manager_model.parameters():
            p.requires_grad_(True)
        trainable = sum(p.numel() for p in manager_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in manager_model.parameters())
        print(f"[MANAGER_GRPO] full_parameter_rl=True trainable_params={trainable}/{total}")
    manager_model.config.use_cache = False
    if not hasattr(manager_model, "warnings_issued") or manager_model.warnings_issued is None:
        manager_model.warnings_issued = {}

    # ---- GRPO config ----
    _grpo_extra: Dict[str, Any] = {}
    if cfg.clip_epsilon_high > 0:
        # DAPO Clip-Higher: asymmetric ratio clipping [1-epsilon, 1+epsilon_high]
        # Allows larger updates when increasing probability of under-explored actions
        # without symmetrically loosening the lower bound that guards against collapse.
        _grpo_extra["epsilon_high"] = float(cfg.clip_epsilon_high)
    if cfg.generation_batch_size > 0:
        if "generation_batch_size" not in inspect.signature(GRPOConfig.__init__).parameters:
            raise RuntimeError(
                "The installed TRL does not support generation_batch_size. "
                "Upgrade TRL before using --mgr_generation_batch_size."
            )
        if cfg.generation_batch_size % cfg.num_generations != 0:
            raise ValueError(
                "generation_batch_size must be divisible by num_generations; "
                f"got {cfg.generation_batch_size} and {cfg.num_generations}."
            )
        _grpo_extra["generation_batch_size"] = int(cfg.generation_batch_size)
    # Budget-truncated completions (no EOS) carry no usable learning signal and
    # their rewards are dominated by the truncation, not the policy; mask them
    # from the loss so budget events stop acting as anti-tool gradients.
    if "mask_truncated_completions" in inspect.signature(GRPOConfig.__init__).parameters:
        _grpo_extra["mask_truncated_completions"] = True
    else:
        print(
            "[MANAGER_GRPO] WARNING: installed TRL lacks mask_truncated_completions; "
            "budget-truncated rollouts will enter the loss. Upgrade TRL."
        )
    grpo_args = GRPOConfig(
        output_dir=cfg.out_dir,
        remove_unused_columns=False,
        max_completion_length=int(cfg.max_completion_length),
        temperature=float(cfg.temperature),
        top_p=float(cfg.top_p),
        top_k=int(cfg.top_k),
        min_p=float(cfg.min_p),
        num_generations=int(cfg.num_generations),
        learning_rate=float(cfg.learning_rate),
        bf16=(device == "cuda"),
        beta=float(cfg.grpo_beta),
        scale_rewards=str(cfg.scale_rewards),
        report_to=(["wandb"] if cfg.use_wandb else []),
        use_vllm=False,
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        max_tool_calling_iterations=3,           # we allow up to 3 tools
        chat_template_kwargs={"enable_thinking": bool(cfg.enable_thinking)},
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=None,
        log_unique_prompts=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
        **_grpo_extra,
    )

    reward_funcs = build_reward_funcs(
        fail_buffer_jsonl=fail_buffer,
        raw_trace_jsonl=raw_trace,
        routing_efficiency_bonus=cfg.routing_efficiency_bonus,
        tool_use_bonus=cfg.tool_use_bonus,
        ccr_mode=cfg.ccr_mode,
        ccr_p_high=cfg.ccr_p_high,
        ccr_p_low=cfg.ccr_p_low,
        ccr_k_max=cfg.ccr_k_max,
        adc_mode=cfg.adc_mode,
        adc_cost_per_tool=cfg.adc_cost_per_tool,
        adc_draft_bonus=cfg.adc_draft_bonus,
        adc_missing_draft_penalty=cfg.adc_missing_draft_penalty,
        adc_final_bonus=cfg.adc_final_bonus,
        adc_variant=cfg.adc_variant,
        adc_format_penalty=cfg.adc_format_penalty,
        adc_cost_warmup_steps=cfg.adc_cost_warmup_steps,
        is_main_process=is_main,
    )
    if cfg.adc_mode:
        print(
            f"[MANAGER_GRPO] ADC mode ON  variant={cfg.adc_variant}  "
            f"draft_bonus={cfg.adc_draft_bonus}  "
            f"missing_draft_penalty={cfg.adc_missing_draft_penalty}  "
            f"final_bonus={cfg.adc_final_bonus}  "
            f"cost_per_tool={cfg.adc_cost_per_tool}  "
            f"cost_warmup_steps={cfg.adc_cost_warmup_steps}  "
            f"format_penalty={cfg.adc_format_penalty}  "
            f"scale_rewards={cfg.scale_rewards}"
        )
        if cfg.scale_rewards == "group":
            print(
                "[MANAGER_GRPO] WARNING: scale_rewards='group' with ADC amplifies "
                "per-tool cost differences in all-correct groups into full-size "
                "negative advantages -> tool-use collapse. Use 'none' or 'batch'."
            )
        if cfg.adc_variant != "anytime":
            print(
                f"[MANAGER_GRPO] WARNING: adc_variant={cfg.adc_variant} is a provably "
                f"exploitable reward — use only as an RQ3 ablation arm."
            )
    elif cfg.ccr_mode:
        print(
            f"[MANAGER_GRPO] CCR mode ON  p_high={cfg.ccr_p_high} "
            f"p_low={cfg.ccr_p_low} k_max={cfg.ccr_k_max}"
        )
    else:
        print(
            f"[MANAGER_GRPO] binary reward  "
            f"routing_efficiency_bonus={cfg.routing_efficiency_bonus} "
            f"tool_use_bonus={cfg.tool_use_bonus}"
        )

    if binding_mode == "environment":
        trainer = GRPOTrainer(
            model=manager_model,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=manager_tok,
            reward_funcs=reward_funcs,
            rollout_func=None,
            environment_factory=ManagerToolEnvironment,
        )
    else:
        trainer = GRPOTrainer(
            model=manager_model,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=manager_tok,
            reward_funcs=reward_funcs,
            rollout_func=None,
            tools=[extractor_tool, reasoner_tool, verifier_tool],
        )

    trainer.train()
    trainer.model.save_pretrained(cfg.out_dir)
    manager_tok.save_pretrained(cfg.out_dir)

    subagent_keys = (
        sorted(pool._kinds)
        if isinstance(pool, RemoteSubagentPool)
        else sorted(pool._agents.keys())
    )
    write_json(os.path.join(cfg.out_dir, "manager_run_config.json"), {
        "base_model": cfg.base_model,
        "binding_mode": binding_mode,
        "n_train_rows": len(cfg.rows),
        "subagents": subagent_keys,
        "subagent_max_new_tokens": {
            kind: default_subagent_max_new_tokens(kind) for kind in subagent_keys
        },
        "manager_adapter": cfg.manager_adapter,
        "full_parameter_rl": bool(cfg.full_parameter_rl),
        "enable_thinking": bool(cfg.enable_thinking),
        "max_completion_length": int(cfg.max_completion_length),
        "temperature": float(cfg.temperature),
        "top_p": float(cfg.top_p),
        "top_k": int(cfg.top_k),
        "min_p": float(cfg.min_p),
        "learning_rate": float(cfg.learning_rate),
        "generation_batch_size": int(cfg.generation_batch_size),
        "subagent_server_url": cfg.subagent_server_url or "",
        "routing_efficiency_bonus": cfg.routing_efficiency_bonus,
        "tool_use_bonus": cfg.tool_use_bonus,
        "ccr_mode": bool(cfg.ccr_mode),
        "ccr_p_high": float(cfg.ccr_p_high),
        "ccr_p_low": float(cfg.ccr_p_low),
        "ccr_k_max": int(cfg.ccr_k_max),
        "adc_mode": bool(cfg.adc_mode),
        "adc_cost_per_tool": float(cfg.adc_cost_per_tool),
        "adc_draft_bonus": float(cfg.adc_draft_bonus),
        "adc_missing_draft_penalty": float(cfg.adc_missing_draft_penalty),
        "adc_final_bonus": float(cfg.adc_final_bonus),
        "adc_variant": str(cfg.adc_variant),
        "adc_format_penalty": float(cfg.adc_format_penalty),
        "adc_cost_warmup_steps": int(cfg.adc_cost_warmup_steps),
        "scale_rewards": str(cfg.scale_rewards),
        "mask_truncated_completions": bool(_grpo_extra.get("mask_truncated_completions", False)),
    })
    print(f"[MANAGER_GRPO] saved -> {cfg.out_dir}")
