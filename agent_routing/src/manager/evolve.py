"""Evolve loop: turn manager failures into new SFT data, then SFT-train manager.

Three steps (called separately or together via pipeline.stages):
  1. build_manager_sft_from_failures: read fail_buffer.jsonl, run subagents
     and an optional teacher to construct multi-turn SFT trajectories.
  2. train_manager_sft: do per-turn SFT on the constructed jsonl.
  3. (back to GRPO with the SFT'd model as init — handled at pipeline level)

The teacher's job here is to PICK A TOOL SEQUENCE (0-3 tools) for each failed
example. The teacher does NOT generate the final answer text; we use the
ground truth to construct the final ANSWER_<TOKEN> line.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ..benchmarks.base import StandardRow, question_hash as _question_hash
from ..subagents.runtime import FrozenSubagent, RemoteSubagentPool, SubagentPool
from ..teachers.base import TeacherClient
from ..utils.io import read_jsonl, write_jsonl, write_json
from ..utils.modeling import discover_lora_target_modules, load_text_causal_lm
from ..utils.seed import set_seed

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

from ..utils.chat_template import ensure_nothink_chat_template
from .prompt import (
    build_manager_system_prompt,
    build_manager_tool_schemas,
    build_manager_user_message,
    parse_final_answer,
)

try:
    import requests as _requests
except ImportError:
    _requests = None


_ALLOWED_TOOLS = ("extractor_tool", "reasoner_tool", "verifier_tool")
_TOOL_NAME_TO_KIND = {
    "extractor_tool": "extractor",
    "reasoner_tool": "reasoner",
    "verifier_tool": "verifier",
}


def _label_to_token(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _final_answer_str(gt: str) -> str:
    return f"ANSWER_{_label_to_token(gt)}"


def _teacher_choose_tool_sequence(
    teacher: Optional[TeacherClient],
    question: str,
    context: str,
    choices: Dict[str, str],
    available_kinds: List[str],
    current_draft: str = "",
    task_description: str = "",
    fallback_seq: Optional[List[str]] = None,
) -> List[str]:
    """Ask the teacher which tool sequence (length 0-3) would best help solve this.

    The teacher does NOT see the GT here — we want it to recommend a sequence
    that a confused-on-this-example manager should follow.

    Returns: list of tool names from _ALLOWED_TOOLS, deduplicated, length<=3.
    """
    available_tools = [k + "_tool" for k in available_kinds if (k + "_tool") in _ALLOWED_TOOLS]

    if teacher is None:
        if fallback_seq is not None:
            return [t for t in fallback_seq if t in available_tools][:3]
        # Heuristic: long context -> extractor first; MCQ -> reasoner; default reasoner.
        seq: List[str] = []
        if context and len(context) > 800 and "extractor_tool" in available_tools:
            seq.append("extractor_tool")
        if "reasoner_tool" in available_tools:
            seq.append("reasoner_tool")
        return seq[:3]

    sys_msg = (
        "You design tool-use plans for a manager agent.\n"
        f"Task: {task_description or 'multiple-choice question answering'}.\n"
        f"Available tools: {available_tools}.\n"
        "Choose a sequence of 0 to 3 tools (no repeats) to create DIVERSE, HIGH-QUALITY training data.\n"
        "Guidelines:\n"
        "  - extractor_tool: use when the question has dense context, complex wording, or requires isolating key facts.\n"
        "  - reasoner_tool: use when multi-step inference or per-choice analysis is needed.\n"
        "  - verifier_tool: use when domain principles matter or reasoning errors are likely.\n"
        "  - 0 tools: only for trivially obvious questions where a manager could answer confidently without any help.\n"
        "  - 3 tools: for hard questions involving specialized knowledge, multi-step reasoning, AND verification risk.\n"
        "Target mix across many examples: ~10% k=0, ~25% k=1, ~45% k=2, ~20% k=3.\n"
        "Return ONLY JSON: {\"tool_sequence\": [\"tool_a\", \"tool_b\"]}"
    )
    choices_block = ""
    if choices:
        lines = [f"  {k}. {v}" for k, v in choices.items()]
        choices_block = "CHOICES:\n" + "\n".join(lines) + "\n\n"
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"{choices_block}"
        f"CONTEXT:\n{context if context else '(no context)'}\n\n"
        f"MANAGER'S CURRENT DRAFT:\n{current_draft or '(unavailable)'}\n"
    )
    try:
        resp = teacher.chat(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2, max_tokens=200,
        )
        text = resp.text
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e <= s:
            raise ValueError("no JSON in teacher response")
        obj = json.loads(text[s:e + 1])
        seq = obj.get("tool_sequence", [])
        if not isinstance(seq, list):
            raise ValueError("tool_sequence not a list")
        out: List[str] = []
        for item in seq:
            t = str(item).strip()
            if t in available_tools and t not in out:
                out.append(t)
            if len(out) >= 3:
                break
        return out
    except Exception:
        return fallback_seq or (
            ["extractor_tool", "reasoner_tool"]
            if context and len(context) > 800
            else ["reasoner_tool"]
        )


def _tool_call_message(
    tool_name: str,
    eid: int,
    call_id: str,
    binding_mode: str,
    content: str = "",
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {"example_id": int(eid)} if binding_mode == "argument" else {}
    if extra_args:
        args.update(extra_args)
    msg: Dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            # HF chat templates require a mapping here; the Qwen3.5 template
            # raises "Can only get item pairs from a mapping" on the OpenAI
            # wire format (a JSON string). _normalize_tool_args still converts
            # string arguments found in older JSONL rows.
            "function": {"name": tool_name, "arguments": dict(args)},
        }],
    }
    if content:
        msg["content"] = content
    return msg


def _draft_answer_str(gt: str) -> str:
    return f"DRAFT_ANSWER_{_label_to_token(gt)}"


class _RemoteManagerDraftGenerator:
    """Elicit state-dependent manager beliefs from the shared vLLM server.

    The sub-agent server exposes the base model as ``base`` alongside the three
    LoRA sub-agents. This lets cold-start alternate manager -> sub-agent -> manager
    without loading four separate 8B model copies in one Python process.
    """

    def __init__(
        self,
        server_url: str,
        model_name: str = "base",
        max_new_tokens: int = 256,
        timeout: int = 180,
    ) -> None:
        if _requests is None:
            raise RuntimeError("requests is required for stepwise cold-start drafts")
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def predict(
        self,
        messages: List[Dict[str, Any]],
        choice_keys: List[str],
    ) -> Optional[str]:
        if not messages:
            return None
        calibration = (
            "\n\nCalibration pass: tools are temporarily unavailable. Based only on "
            "the conversation above, report your current best answer. Put your answer "
            "on the FIRST line as exactly one ANSWER_<TOKEN> line, then reasoning may follow."
        )
        request_messages = [dict(m) for m in messages]
        request_messages = [dict(m) for m in messages]
        # tool_call arguments must be a JSON *string* for the vLLM OpenAI API
        for m in request_messages:
            if m.get("tool_calls"):
                m["tool_calls"] = [
                    {**tc, "function": {
                        **tc["function"],
                        "arguments": tc["function"]["arguments"] if isinstance(tc["function"].get("arguments"), str)
                                     else __import__("json").dumps(tc["function"].get("arguments") or {})
                    }} for tc in m["tool_calls"]
                ]
        request_messages[0] = dict(request_messages[0])
        request_messages[0]["content"] = str(request_messages[0].get("content") or "") + calibration
        request_messages[0] = dict(request_messages[0])
        request_messages[0]["content"] = str(request_messages[0].get("content") or "") + calibration
        # NOTE: this is a raw JSON payload, not an openai-python call —
        # "extra_body" is a client-library concept and must not appear here.
        # vLLM reads chat_template_kwargs at the top level.
        payload = {
            "model": self.model_name,
            "messages": request_messages,
            "temperature": 0.0,
            "max_tokens": self.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = _requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = str(resp.json()["choices"][0]["message"].get("content") or "")
        return parse_final_answer(text, choice_keys)


def _build_remote_draft_generator(cfg: "ColdStartSFTConfig") -> Optional[_RemoteManagerDraftGenerator]:
    if cfg.draft_source != "base_stepwise":
        return None
    if not cfg.draft_server_url:
        raise ValueError(
            "base_stepwise cold-start requires --coldstart_draft_server_url. "
            "Start scripts/start_subagent_server.sh first."
        )
    return _RemoteManagerDraftGenerator(
        server_url=cfg.draft_server_url,
        model_name=cfg.draft_model_name,
        max_new_tokens=cfg.draft_max_new_tokens,
    )


def _predict_base_initial_drafts(
    base_model: str,
    rows: List[StandardRow],
    binding_mode: str,
    task_description: str,
    seed: int,
    max_new_tokens: int = 256,
) -> Dict[int, str]:
    """Greedily elicit the base manager's pre-tool answer for each example.

    Cold-start previously copied ``row.ground_truth`` into every draft.  That
    produced oracle belief trajectories and made the verifier see only correct
    candidates.  This pass deliberately runs *before* subagents are loaded, so
    an 8B setup does not need the base manager and three sub-agent models resident
    at the same time.

    Only parseable on-policy predictions are returned.  We never fall back to
    the ground truth: a format failure is skipped and reported instead of
    silently reintroducing privileged information.
    """
    if not rows:
        return {}

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    ensure_nothink_chat_template(tok, tag="COLDSTART_DRAFTS")
    model = load_text_causal_lm(
        base_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    predictions: Dict[int, str] = {}
    for row in rows:
        keys = list(row.choices.keys())
        answer_lines = "\n".join(f"ANSWER_{_label_to_token(k)}" for k in keys)
        system = (
            (task_description or "Solve the multiple-choice question.")
            + "\nTools are unavailable in this calibration pass. Put your answer on the "
              "FIRST line as exactly one of the following, then reasoning may follow:\n"
            + answer_lines
        )
        user = build_manager_user_message(
            example_id=int(row.example_id),
            question=row.question,
            context=row.context,
            choices=row.choices,
            binding_mode=binding_mode,
        )
        prompt = _render_chat(
            tok,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        text = tok.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_final_answer(text, keys)
        if pred is not None:
            predictions[int(row.example_id)] = pred

    # Release the base model before the three frozen sub-agents are loaded.
    del model
    del tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def _coldstart_draft_map(
    cfg: "ColdStartSFTConfig",
    rows: List[StandardRow],
) -> Dict[int, str]:
    if cfg.draft_source == "oracle":
        return {int(r.example_id): str(r.ground_truth) for r in rows}
    if cfg.draft_source == "base_stepwise":
        # Generated from the live conversation inside the trajectory builder.
        return {}
    if cfg.draft_source != "base_initial":
        raise ValueError(f"Unknown cold-start draft source: {cfg.draft_source}")
    return _predict_base_initial_drafts(
        base_model=cfg.base_model,
        rows=rows,
        binding_mode=cfg.binding_mode,
        task_description=cfg.task_description,
        seed=cfg.seed,
        max_new_tokens=cfg.draft_max_new_tokens,
    )


@dataclass
class EvolveSFTConfig:
    base_model: str
    extractor_adapter: Optional[str]
    reasoner_adapter: Optional[str]
    verifier_adapter: Optional[str]
    rows: List[StandardRow]
    fail_buffer_jsonl: str
    out_dir: str
    teacher: Optional[TeacherClient] = None
    seed: int = 42
    max_fail_samples: int = 1500
    binding_mode: str = "environment"
    task_description: str = ""


def _register_available_subagents(
    base_model: str,
    extractor_adapter: Optional[str],
    reasoner_adapter: Optional[str],
    verifier_adapter: Optional[str],
    device: str,
) -> tuple[SubagentPool, List[str]]:
    pool = SubagentPool()
    available_kinds: List[str] = []
    if extractor_adapter:
        pool.register(FrozenSubagent(base_model, extractor_adapter, "extractor", device))
        available_kinds.append("extractor")
    if reasoner_adapter:
        pool.register(FrozenSubagent(base_model, reasoner_adapter, "reasoner", device))
        available_kinds.append("reasoner")
    if verifier_adapter:
        pool.register(FrozenSubagent(base_model, verifier_adapter, "verifier", device))
        available_kinds.append("verifier")
    return pool, available_kinds


def _coldstart_pool(
    cfg: "ColdStartSFTConfig",
    device: str,
) -> tuple[Any, List[str]]:
    if cfg.subagent_server_url:
        kinds = ["extractor", "reasoner", "verifier"]
        return RemoteSubagentPool(cfg.subagent_server_url, registered_kinds=kinds), kinds
    return _register_available_subagents(
        cfg.base_model,
        cfg.extractor_adapter,
        cfg.reasoner_adapter,
        cfg.verifier_adapter,
        device,
    )


def _coldstart_fallback_sequence(idx: int, context: str, available_kinds: List[str]) -> List[str]:
    available_tools = {k + "_tool" for k in available_kinds}
    if context and len(context) > 800 and "extractor_tool" in available_tools:
        seq = ["extractor_tool", "reasoner_tool"]
    elif idx % 5 == 0 and "extractor_tool" in available_tools:
        seq = ["extractor_tool", "reasoner_tool"]
    elif idx % 5 == 1 and "verifier_tool" in available_tools:
        seq = ["verifier_tool", "reasoner_tool"]
    else:
        seq = ["reasoner_tool"]
    return [t for t in seq if t in available_tools][:3]


def _build_manager_tool_sft_rows(
    rows: List[StandardRow],
    pool: SubagentPool,
    available_kinds: List[str],
    teacher: Optional[TeacherClient],
    draft_answers: Dict[int, str],
    draft_source: str,
    draft_generator: Optional[_RemoteManagerDraftGenerator],
    forced_sequences: Optional[Dict[int, List[str]]],
    binding_mode: str,
    task_description: str,
    cache_namespace: str,
) -> List[Dict[str, Any]]:
    try:
        from tqdm import tqdm
        _iter = tqdm(rows, desc=f"[{cache_namespace}] building SFT rows", unit="ex")
    except ImportError:
        _iter = rows

    sft_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for idx, row in enumerate(_iter):
        eid = int(row.example_id)
        sys_prompt = build_manager_system_prompt(
            label_keys=list(row.choices.keys()),
            task_description=task_description,
        )
        user_msg = build_manager_user_message(
            example_id=eid,
            question=row.question,
            context=row.context,
            choices=row.choices,
            binding_mode=binding_mode,
        )
        base_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        draft_answer = draft_answers.get(eid)
        if draft_answer not in row.choices and draft_generator is not None:
            draft_answer = draft_generator.predict(base_messages, list(row.choices.keys()))
        if draft_answer not in row.choices:
            # Never substitute ground truth for a missing/unparseable manager
            # prediction. Skipping exposes format failures instead of creating
            # an oracle belief trajectory.
            continue

        fallback_seq = _coldstart_fallback_sequence(idx, row.context, available_kinds)
        if forced_sequences is not None and eid in forced_sequences:
            seq = [t for t in forced_sequences[eid] if t in _ALLOWED_TOOLS][:3]
        else:
            seq = _teacher_choose_tool_sequence(
                teacher=teacher,
                question=row.question,
                context=row.context,
                choices=row.choices,
                available_kinds=available_kinds,
                current_draft=draft_answer,
                task_description=task_description,
                fallback_seq=fallback_seq,
            )

        final_text = _final_answer_str(row.ground_truth)
        qhash = _question_hash(row.question)
        audit = {
            "draft_source": draft_source,
            "initial_draft": draft_answer,
            "initial_draft_correct": draft_answer == row.ground_truth,
            "ground_truth": row.ground_truth,
        }
        if not seq:
            sft_rows.append({
                "example_id": eid,
                "question_hash": qhash,
                "prompt": base_messages,
                "response": [{"role": "assistant", "content": final_text}],
                "draft_sequence": [draft_answer],
                "final_on_policy_draft": draft_answer,
                **audit,
            })
            continue

        history = list(base_messages)
        # Belief states, not tool turns: initial belief followed by one
        # post-sub-agent belief per executed tool. Keeping the final update is
        # essential for measuring whether a one-tool trajectory corrected or
        # corrupted the manager's answer.
        draft_sequence: List[str] = [draft_answer]
        for i, tname in enumerate(seq):
            kind = _TOOL_NAME_TO_KIND[tname]
            if not pool.has(kind):
                continue
            call_id = f"call_{eid}_{i+1}"
            draft_text = _draft_answer_str(draft_answer)
            # ADC policy: every tool-calling turn states the current draft answer;
            # verifier calls pass the draft so it audits that hypothesis.
            extra_args = {"current_draft": draft_answer} if tname == "verifier_tool" else None
            asst_call = _tool_call_message(
                tname, eid, call_id, binding_mode,
                content=draft_text, extra_args=extra_args,
            )
            sft_rows.append({
                "example_id": eid,
                "question_hash": qhash,
                "prompt": list(history),
                "response": [asst_call],
                **audit,
                "draft_step": i,
                "current_draft": draft_answer,
                "current_draft_correct": draft_answer == row.ground_truth,
            })
            tool_output = pool.call(
                agent_kind=kind,
                example_id=eid,
                question=row.question,
                context=row.context,
                choices=row.choices,
                cache_namespace=cache_namespace,
                candidate_answer=(draft_answer if kind == "verifier" else ""),
            )
            tool_msg = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tname,
                "content": tool_output,
            }
            history = history + [asst_call, tool_msg]
            if draft_generator is not None:
                updated = draft_generator.predict(history, list(row.choices.keys()))
                if updated in row.choices:
                    draft_answer = updated
            draft_sequence.append(draft_answer)

        sft_rows.append({
            "example_id": eid,
            "question_hash": qhash,
            "prompt": list(history),
            # Drafts are required only on delegating turns.  The terminal turn
            # receives ordinary oracle answer supervision without pretending
            # that the manager's pre-tool belief was oracle-correct.
            "response": [{"role": "assistant", "content": final_text}],
            **audit,
            "draft_sequence": draft_sequence,
            "final_on_policy_draft": draft_answer,
        })
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{cache_namespace}] {idx+1}/{len(rows)} examples | {len(sft_rows)} SFT turns | {elapsed:.0f}s elapsed")
    return sft_rows


def build_manager_sft_from_failures(cfg: EvolveSFTConfig) -> str:
    """Read fail buffer, build per-turn SFT trajectories, write to disk.

    Output is a JSONL where each row is a per-turn (prompt, response) pair:
      - turn 1: user message -> first tool_call (or final answer if seq is empty)
      - turn 2: turn1 + tool output -> second tool_call (or final answer)
      - turn 3+: ... up to 3 tools, then final answer turn
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pool, available_kinds = _register_available_subagents(
        cfg.base_model,
        cfg.extractor_adapter,
        cfg.reasoner_adapter,
        cfg.verifier_adapter,
        device,
    )

    row_index = {int(r.example_id): r for r in cfg.rows}

    # Read failures, dedupe by example_id, cap.
    fails: List[int] = []
    failed_drafts: Dict[int, str] = {}
    seen = set()
    if not os.path.exists(cfg.fail_buffer_jsonl):
        raise FileNotFoundError(f"fail_buffer not found: {cfg.fail_buffer_jsonl}")
    for row in read_jsonl(cfg.fail_buffer_jsonl):
        eid = row.get("example_id")
        if eid is None:
            continue
        try:
            eid = int(eid)
        except Exception:
            continue
        if eid in seen:
            continue
        if eid not in row_index:
            continue
        pred = str(row.get("pred") or "").strip()
        if pred not in row_index[eid].choices:
            # A malformed completion has no defensible belief label.  Do not
            # replace it with the ground truth in an evolve trajectory.
            continue
        seen.add(eid)
        fails.append(eid)
        failed_drafts[eid] = pred
        if len(fails) >= cfg.max_fail_samples:
            break

    print(f"[EVOLVE] {len(fails)} unique failed example_ids selected from buffer.")

    selected_rows = [row_index[eid] for eid in fails]
    sft_rows = _build_manager_tool_sft_rows(
        rows=selected_rows,
        pool=pool,
        available_kinds=available_kinds,
        teacher=cfg.teacher,
        draft_answers=failed_drafts,
        draft_source="failed_manager",
        draft_generator=None,
        forced_sequences=None,
        binding_mode=cfg.binding_mode,
        task_description=cfg.task_description,
        cache_namespace="evolve",
    )

    out_path = os.path.join(cfg.out_dir, "manager_sft_from_failures.jsonl")
    write_jsonl(out_path, sft_rows)
    write_json(os.path.join(cfg.out_dir, "evolve_run_config.json"), {
        "n_failed_examples": len(fails),
        "n_sft_rows": len(sft_rows),
        "available_kinds": available_kinds,
        "binding_mode": cfg.binding_mode,
        "teacher_provider": cfg.teacher.provider if cfg.teacher else "heuristic",
        "teacher_model": cfg.teacher.model if cfg.teacher else "",
    })
    print(f"[EVOLVE] wrote {len(sft_rows)} SFT rows -> {out_path}")
    return out_path


@dataclass
class ColdStartSFTConfig:
    base_model: str
    extractor_adapter: Optional[str]
    reasoner_adapter: Optional[str]
    verifier_adapter: Optional[str]
    rows: List[StandardRow]
    out_dir: str
    teacher: Optional[TeacherClient] = None
    seed: int = 42
    n_samples: int = 300
    binding_mode: str = "environment"
    task_description: str = ""
    draft_source: str = "base_stepwise"
    draft_max_new_tokens: int = 256
    draft_server_url: str = ""
    draft_model_name: str = "base"
    subagent_server_url: str = ""
    oracle_cost_per_tool: float = 0.05


def build_manager_sft_from_rows(
    cfg: ColdStartSFTConfig,
    forced_sequences: Optional[Dict[int, List[str]]] = None,
    out_path: Optional[str] = None,
) -> str:
    """Build manager tool-call SFT rows from ordinary training examples."""
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample = list(cfg.rows)
    random.Random(cfg.seed).shuffle(sample)
    if cfg.n_samples > 0:
        sample = sample[:cfg.n_samples]

    # Elicit beliefs before loading sub-agents to avoid keeping four model copies
    # resident at once.  Oracle drafts remain available only as an explicit
    # ablation via draft_source="oracle".
    draft_answers = _coldstart_draft_map(cfg, sample)
    draft_generator = _build_remote_draft_generator(cfg)
    if draft_generator is None:
        sample = [r for r in sample if int(r.example_id) in draft_answers]
    if not sample:
        raise ValueError(
            "No parseable base-manager drafts were produced for cold-start. "
            "Inspect the base model answer format instead of falling back to ground truth."
        )
    pool, available_kinds = _coldstart_pool(cfg, device)
    if not available_kinds:
        raise ValueError("No subagent adapters available for cold-start SFT.")

    n_correct = sum(
        draft_answers.get(int(r.example_id)) == r.ground_truth for r in sample
    ) if draft_answers else 0
    print(
        f"[COLDSTART] building SFT data for {len(sample)} examples | "
        f"draft_source={cfg.draft_source} "
        f"initial_draft_acc={(n_correct/max(1, len(sample)) if draft_answers else float('nan')):.3f} | "
        f"subagents={available_kinds}"
    )
    sft_rows = _build_manager_tool_sft_rows(
        rows=sample,
        pool=pool,
        available_kinds=available_kinds,
        teacher=cfg.teacher,
        draft_answers=draft_answers,
        draft_source=cfg.draft_source,
        draft_generator=draft_generator,
        forced_sequences=forced_sequences,
        binding_mode=cfg.binding_mode,
        task_description=cfg.task_description,
        cache_namespace="coldstart",
    )

    if out_path is None:
        out_path = os.path.join(cfg.out_dir, "manager_sft_coldstart.jsonl")
    seqs = [r["draft_sequence"] for r in sft_rows if "draft_sequence" in r]
    n_changed = sum(1 for s in seqs if len(set(s)) > 1)
    write_jsonl(out_path, sft_rows)
    write_json(out_path + ".meta.json", {
        "n_examples": len(sample),
        "n_sft_rows": len(sft_rows),
        "available_kinds": available_kinds,
        "draft_changed_after_tools": n_changed,          # ← 新增
        "draft_change_rate": round(n_changed / max(1, len(seqs)), 3),  # ← 新增
        "binding_mode": cfg.binding_mode,
        "draft_source": cfg.draft_source,
        "n_parseable_drafts": len(sample),
        "initial_draft_accuracy": (
            n_correct / max(1, len(sample)) if draft_answers else None
        ),
        "teacher_provider": cfg.teacher.provider if cfg.teacher else "heuristic",
        "teacher_model": cfg.teacher.model if cfg.teacher else "",
    })
    print(f"[COLDSTART] wrote {len(sft_rows)} SFT rows from {len(sample)} examples -> {out_path}")
    return out_path


def make_diverse_sequences(
    rows: List[StandardRow],
    available_kinds: List[str],
    seed: int = 42,
) -> Dict[int, List[str]]:
    """Assign a force-balanced tool sequence to each row without calling any model.

    Deterministic given seed. Sequences use only available agent kinds.
    Distribution when all 3 agents present:
      k=0 (direct answer):           35%
      k=1 reasoner only:             25%
      k=1 extractor only:             5%
      k=2 extractor→reasoner:        15%
      k=2 reasoner→verifier:         15%
      k=3 extractor→reasoner→verifier: 5%
    """
    has_ext = "extractor" in available_kinds
    has_rsn = "reasoner" in available_kinds
    has_vrf = "verifier" in available_kinds

    # (weight, sequence) — only include sequences whose agents are available
    _slots = [
        (35, []),
        (25, ["reasoner_tool"] if has_rsn else []),
        (5,  ["extractor_tool"] if has_ext else []),
        (15, ["extractor_tool", "reasoner_tool"] if (has_ext and has_rsn) else (["reasoner_tool"] if has_rsn else [])),
        (15, ["reasoner_tool", "verifier_tool"] if (has_rsn and has_vrf) else (["reasoner_tool"] if has_rsn else [])),
        (5, ["extractor_tool", "reasoner_tool", "verifier_tool"] if (has_ext and has_rsn and has_vrf) else (["reasoner_tool"] if has_rsn else [])),
    ]

    # Build a flat template list proportional to weights
    template: List[List[str]] = []
    for weight, seq in _slots:
        template.extend([seq] * weight)  # total = 100 slots

    rng = random.Random(seed)
    shuffled_rows = list(rows)
    rng.shuffle(shuffled_rows)

    sequences: Dict[int, List[str]] = {}
    for i, row in enumerate(shuffled_rows):
        sequences[int(row.example_id)] = list(template[i % len(template)])

    return sequences


def make_cost_aware_sequences(
    cfg: ColdStartSFTConfig,
    rows: List[StandardRow],
) -> Dict[int, List[str]]:
    """Enumerate sub-agent subsets and imitate the best correctness-cost action.

    Ground truth selects the expert action sequence, but is never inserted into
    a draft. Every simulated draft is elicited from the base manager after the
    actual sub-agent output, making the supervision appropriate for stopping.
    """
    generator = _build_remote_draft_generator(cfg)
    if generator is None:
        raise ValueError("cost-aware oracle sequences require base_stepwise drafts")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool, available_kinds = _coldstart_pool(cfg, device)
    tools = [k + "_tool" for k in available_kinds]
    candidates: List[List[str]] = [[]]
    candidates.extend([[t] for t in tools])
    candidates.extend([
        ["extractor_tool", "reasoner_tool"],
        ["extractor_tool", "verifier_tool"],
        ["reasoner_tool", "verifier_tool"],
        ["extractor_tool", "reasoner_tool", "verifier_tool"],
    ])
    candidates = [seq for seq in candidates if all(t in tools for t in seq)]

    selected: Dict[int, List[str]] = {}
    for row in rows:
        eid = int(row.example_id)
        system = build_manager_system_prompt(
            label_keys=list(row.choices.keys()),
            task_description=cfg.task_description,
        )
        user = build_manager_user_message(
            example_id=eid,
            question=row.question,
            context=row.context,
            choices=row.choices,
            binding_mode=cfg.binding_mode,
        )
        base_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        initial = generator.predict(base_messages, list(row.choices.keys()))
        if initial not in row.choices:
            continue

        best_seq: List[str] = []
        best_utility = float("-inf")
        for seq in candidates:
            history = list(base_messages)
            current = initial
            for i, tname in enumerate(seq):
                kind = _TOOL_NAME_TO_KIND[tname]
                call_id = f"oracle_{eid}_{i+1}"
                asst = _tool_call_message(
                    tname,
                    eid,
                    call_id,
                    cfg.binding_mode,
                    content=_draft_answer_str(current),
                    extra_args=(
                        {"current_draft": current}
                        if tname == "verifier_tool" else None
                    ),
                )
                output = pool.call(
                    agent_kind=kind,
                    example_id=eid,
                    question=row.question,
                    context=row.context,
                    choices=row.choices,
                    cache_namespace="coldstart_oracle",
                    candidate_answer=(current if kind == "verifier" else ""),
                )
                history.extend([asst, {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tname,
                    "content": output,
                }])
                updated = generator.predict(history, list(row.choices.keys()))
                if updated in row.choices:
                    current = updated

            utility = (1.0 if current == row.ground_truth else 0.0) - (
                cfg.oracle_cost_per_tool * len(seq)
            )
            if utility > best_utility or (
                utility == best_utility and len(seq) < len(best_seq)
            ):
                best_utility = utility
                best_seq = list(seq)
        selected[eid] = best_seq
    return selected


def build_manager_sft_from_sequences(
    cfg: ColdStartSFTConfig,
    sequences: Dict[int, List[str]],
    out_path: Optional[str] = None,
) -> str:
    """Build cold-start trajectories with externally selected tool sequences.

    Draft generation is controlled by cfg.draft_source; with the default
    base_stepwise mode, every tool result is followed by a fresh base-manager
    belief elicitation before another delegation.
    """
    return build_manager_sft_from_rows(
        cfg,
        forced_sequences=sequences,
        out_path=out_path or os.path.join(cfg.out_dir, "coldstart_from_sequences_sft.jsonl"),
    )

# -------------- Manager SFT --------------

@dataclass
class ManagerSFTConfig:
    base_model: str
    train_jsonl: str
    out_dir: str
    seed: int = 42
    max_seq_len: int = 8192
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = -1
    bf16: bool = True
    # Render the tool schemas into the SFT prompts so the SFT distribution
    # matches GRPO/eval rollouts, where the chat template injects a "# Tools"
    # section. Training without it teaches tool calls the model never sees
    # advertised at rollout time.
    binding_mode: str = "environment"
    include_tool_schemas: bool = True

def _normalize_tool_args(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """HF chat templates want tool_call arguments as a dict; the OpenAI API wants a JSON string."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                tc["function"] = fn
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out
def _render_chat(tokenizer, messages, add_generation_prompt: bool, tools=None) -> str:
    messages = _normalize_tool_args(messages)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            tools=tools, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            tools=tools,
        )


def _tokenize_manager_sft(rows: List[Dict[str, Any]], tok, max_seq_len: int, tools=None) -> Dataset:
    eos = tok.eos_token or ""
    prefix_mismatches = 0

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal prefix_mismatches
        prompt_msgs = ex["prompt"]
        response_msgs = ex["response"]
        if isinstance(response_msgs, dict):
            response_msgs = [response_msgs]
        elif isinstance(response_msgs, str):
            response_msgs = [{"role": "assistant", "content": response_msgs}]

        prompt_text = _render_chat(tok, prompt_msgs, add_generation_prompt=True, tools=tools)
        full_text = _render_chat(
            tok, prompt_msgs + response_msgs, add_generation_prompt=False, tools=tools
        )
        # Chat templates already close the assistant turn with EOS (e.g.
        # "<|im_end|>\n"); appending another EOS would teach the model to emit
        # a duplicate end-of-turn token.
        if eos and eos not in full_text[-(len(eos) + 8):]:
            full_text = full_text + eos

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = tok(full_text, add_special_tokens=False)
        # The label mask assumes token-level prefix identity between the
        # rendered prompt and the full conversation. Verify instead of hoping:
        # a mismatched boundary silently trains on (or masks) the wrong span.
        plen = len(prompt_ids)
        if full["input_ids"][:plen] != prompt_ids:
            common = 0
            for a, b in zip(full["input_ids"], prompt_ids):
                if a != b:
                    break
                common += 1
            plen = common
            prefix_mismatches += 1
        input_ids = full["input_ids"][:max_seq_len]
        attention_mask = full["attention_mask"][:max_seq_len]
        plen = min(plen, max_seq_len)
        labels = ([-100] * plen) + input_ids[plen:]
        labels = labels[:max_seq_len]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = Dataset.from_list(rows)
    mapped = ds.map(_map, remove_columns=ds.column_names)
    if prefix_mismatches:
        print(
            f"[MANAGER_SFT] WARNING: {prefix_mismatches} rows had a "
            f"prompt/full tokenization boundary mismatch; label masks were "
            f"realigned to the longest common token prefix."
        )
    return mapped


def train_manager_sft(cfg: ManagerSFTConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    # Match GRPO/eval: normalize the template default to nothink so the SFT
    # target format is exactly what the rollout-time parser expects.
    ensure_nothink_chat_template(tok, tag="MANAGER_SFT")

    dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
    model = load_text_causal_lm(
        cfg.base_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.config.use_cache = False

    if cfg.use_lora and PEFT_AVAILABLE:
        target = discover_lora_target_modules(model)
        lconf = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target,
        )
        model = get_peft_model(model, lconf)
        print(f"[MANAGER_SFT/LoRA] r={cfg.lora_r} alpha={cfg.lora_alpha} target_modules={target}")

    rows = read_jsonl(cfg.train_jsonl)
    if not rows:
        raise ValueError(f"No rows in {cfg.train_jsonl}")
    tools = (
        build_manager_tool_schemas(cfg.binding_mode)
        if cfg.include_tool_schemas else None
    )
    print(
        f"[MANAGER_SFT] tokenizing {len(rows)} rows "
        f"(tool schemas in prompt: {bool(tools)}, binding={cfg.binding_mode}) ..."
    )
    train_ds = _tokenize_manager_sft(rows, tok, cfg.max_seq_len, tools=tools)
    total_steps = (len(train_ds) // (cfg.per_device_batch_size * cfg.gradient_accumulation_steps)) * cfg.num_train_epochs
    if cfg.max_steps > 0:
        total_steps = min(total_steps, cfg.max_steps)
    print(f"[MANAGER_SFT] {len(train_ds)} train examples | ~{total_steps} steps | lr={cfg.learning_rate} | epochs={cfg.num_train_epochs}")
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=1,
        save_strategy="epoch",
        bf16=(cfg.bf16 and device == "cuda"),
        fp16=False,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()
    os.makedirs(cfg.out_dir, exist_ok=True)
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"[MANAGER_SFT] saved -> {cfg.out_dir}")
