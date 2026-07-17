"""Stage functions wrapping each major step of the pipeline.

Design:
  - Each `run_*` is a thin orchestrator that takes a StageContext + a few
    explicit args and returns a small result dict (paths produced, stats).
  - The CLI maps argparse flags to these calls.
  - Output paths are auto-namespaced by teacher_id so different teachers'
    artifacts never collide. This is the core enabler of the comparison
    experiment.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..benchmarks.base import StandardRow, question_hash
from ..benchmarks.gpqa import load_gpqa
from ..benchmarks.legalbench import load_legalbench
from ..benchmarks.medqa import load_medqa
from ..benchmarks.mmlu_pro import load_mmlu_pro
from ..manager.prompt import (
    build_manager_system_prompt,
    build_manager_user_message,
    parse_draft_answer,
    parse_final_answer,
)
from ..manager import stopping as stoplib
from ..subagents.prompts.extractor import build_extractor_synth_prompt
from ..subagents.prompts.reasoner import build_reasoner_synth_prompt
from ..subagents.prompts.verifier import build_verifier_synth_prompt
from ..subagents.prompts.runtime_prompts import build_runtime_messages
from ..teachers.base import TeacherClient, build_teacher_client
from ..utils.cache import TeacherCallCache
from ..utils.io import read_jsonl, write_json, write_jsonl
from ..utils.leakage import LeakageAuditor
from ..utils.seed import set_seed


# --------------------- Context ---------------------

@dataclass
class StageContext:
    """Shared paths and configuration across stages."""
    base_model: str
    teacher_id: str                       # e.g. "mmlu_pro_gpt54", used for manager/eval paths
    teacher_provider: str = ""            # filled when a teacher is built
    teacher_model: str = ""
    output_root: str = "outputs"
    seed: int = 42
    binding_mode: str = "auto"
    subagent_teacher_id: str = ""        # if set, subagent adapters come from this id instead of teacher_id

    # Auto-derived sub-roots
    sft_data_root: str = field(init=False)
    adapter_root: str = field(init=False)
    manager_root: str = field(init=False)
    cache_dir: str = field(init=False)
    eval_root: str = field(init=False)

    def __post_init__(self) -> None:
        teacher_slug = self._slug(self.teacher_id)
        adapter_slug = self._slug(self.subagent_teacher_id or self.teacher_id)
        self.sft_data_root = os.path.join(self.output_root, "sft_data", teacher_slug)
        self.adapter_root = os.path.join(self.output_root, "adapters", adapter_slug)
        self.manager_root = os.path.join(self.output_root, "manager", teacher_slug)
        self.cache_dir = os.path.join(self.output_root, "teacher_cache", teacher_slug)
        self.eval_root = os.path.join(self.output_root, "eval", teacher_slug)
        for p in (self.sft_data_root, self.adapter_root, self.manager_root,
                  self.cache_dir, self.eval_root):
            os.makedirs(p, exist_ok=True)

    @staticmethod
    def _slug(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
        return s.strip("_") or "unnamed"

    def adapter_path(self, kind: str) -> str:
        return os.path.join(self.adapter_root, f"{kind}_adapter")

    def sft_jsonl_path(self, kind: str) -> str:
        return os.path.join(self.sft_data_root, f"{kind}_sft.jsonl")

    def sft_log_path(self, kind: str) -> str:
        return os.path.join(self.sft_data_root, f"{kind}_synth_log.jsonl")

    def manager_grpo_dir(self) -> str:
        return os.path.join(self.manager_root, "grpo")

    def manager_coldstart_dir(self) -> str:
        return os.path.join(self.manager_root, "sft_coldstart")

    def manager_sft_dir(self) -> str:
        return os.path.join(self.manager_root, "sft_evolved")

    def evolve_dir(self) -> str:
        return os.path.join(self.manager_root, "evolve")

    def fail_buffer_path(self) -> str:
        return os.path.join(self.manager_grpo_dir(), "fail_buffer.jsonl")


# --------------------- Helpers ---------------------

def _agent_kind_value(agent_kind: Any) -> str:
    return str(getattr(agent_kind, "value", agent_kind)).strip()


def _build_local_teacher_prompt(
    agent_kind: Any,
    row: StandardRow,
    candidate_answer: str = "",
) -> List[Dict[str, str]]:
    kind = _agent_kind_value(agent_kind)
    if kind == "extractor":
        return build_extractor_synth_prompt(row.question, row.context, row.choices)
    if kind == "reasoner":
        return build_reasoner_synth_prompt(row.question, row.context, row.choices)
    if kind == "verifier":
        return build_verifier_synth_prompt(
            row.question, row.context, row.choices, candidate_answer=candidate_answer
        )
    raise ValueError(f"Unknown agent_kind: {agent_kind}")

def _build_teacher(provider: str, model: str, ctx: StageContext) -> TeacherClient:
    teacher = build_teacher_client(provider=provider, model=model)
    ctx.teacher_provider = teacher.provider
    ctx.teacher_model = teacher.model
    return teacher


def _split_rows(
    rows: List[StandardRow],
    train_size: int,
    dev_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List[StandardRow], List[StandardRow], List[StandardRow]]:
    """Honor existing splits when present; otherwise random-split."""
    by_split: Dict[str, List[StandardRow]] = {"train": [], "dev": [], "test": [], "": []}
    for r in rows:
        by_split.setdefault(r.split or "", []).append(r)

    # Any explicit train label wins: a train-only cache (e.g. the GPQA train
    # split built by scripts/build_gpqa_splits.py) must never have a phantom
    # test set carved out of its training rows by the random path below.
    have_explicit = bool(by_split["train"])
    if train_size == 0 and not by_split["train"]:
        # Eval-only pool (e.g. GPQA-Diamond or MMLU-Pro used as zero-shot
        # probes with --train_size 0): honor the loader's split labels.
        # The random path below would set n_test = min(test_size, n//4) and
        # silently shrink a 198-question probe to 49 rows.
        train = []
        dev = list(by_split["dev"])
        test = list(by_split["test"]) or list(by_split[""])
    elif have_explicit:
        train = by_split["train"]
        dev = by_split["dev"]
        test = by_split["test"] or by_split["dev"]
        if not dev:
            # No explicit dev split: carve dev from the train tail rather than
            # aliasing test (dev ⊂ test would leak eval rows into any
            # dev-driven decision).
            n_dev = min(max(dev_size, 0), len(train) // 5)
            if n_dev > 0:
                dev = train[-n_dev:]
                train = train[:-n_dev]
            else:
                dev = []
    else:
        rng = random.Random(seed)
        all_rows = list(rows)
        rng.shuffle(all_rows)
        n = len(all_rows)
        n_test = min(test_size, n // 4)
        n_dev = min(dev_size, (n - n_test) // 4)
        test = all_rows[:n_test]
        dev = all_rows[n_test:n_test + n_dev]
        train = all_rows[n_test + n_dev:]

    if train_size > 0 and len(train) > train_size:
        train = train[:train_size]
    if dev_size > 0 and len(dev) > dev_size:
        dev = dev[:dev_size]
    if test_size > 0 and len(test) > test_size:
        test = test[:test_size]
    return train, dev, test


# --------------------- Stage: data loading ---------------------

def run_load_medqa(
    source: str = "hf",
    hf_dataset: str = "GBaker/MedQA-USMLE-4-options",
    local_path: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    cache_normalized_path: Optional[str] = None,
) -> List[StandardRow]:
    rows = load_medqa(
        source=source, hf_dataset=hf_dataset,
        local_path=local_path, hf_cache_dir=hf_cache_dir,
        max_examples=max_examples,
    )
    print(f"[LOAD_MEDQA] loaded {len(rows)} rows from {source}")
    if cache_normalized_path:
        write_jsonl(cache_normalized_path, [r.to_dict() for r in rows])
        print(f"[LOAD_MEDQA] cached normalized rows -> {cache_normalized_path}")
    return rows


def run_load_legalbench(
    dataset_name: str = "nguha/legalbench",
    configs: str = "",
    split: str = "test",
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    max_labels: int = 12,
    cache_normalized_path: Optional[str] = None,
) -> List[StandardRow]:
    rows, meta = load_legalbench(
        dataset_name=dataset_name,
        configs=configs,
        split=split,
        cache_dir=hf_cache_dir,
        max_examples=max_examples,
        max_labels=max_labels,
    )
    print(f"[LOAD_LEGALBENCH] loaded {len(rows)} rows from {dataset_name} split={split}")
    if meta.get("skipped"):
        print(f"[LOAD_LEGALBENCH] skipped {len(meta['skipped'])} configs")
    if cache_normalized_path:
        write_jsonl(cache_normalized_path, [r.to_dict() for r in rows])
        write_json(cache_normalized_path + ".meta.json", meta)
        print(f"[LOAD_LEGALBENCH] cached normalized rows -> {cache_normalized_path}")
    return rows


# --------------------- Stage: GPQA loading ---------------------

def run_load_gpqa(
    dataset_name: str = "Idavidrein/gpqa",
    subsets: str = "gpqa_diamond",
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    answer_seed: int = 42,
    cache_normalized_path: Optional[str] = None,
    exclude_subsets: str = "",
) -> List[StandardRow]:
    rows = load_gpqa(
        dataset_name=dataset_name,
        subsets=subsets,
        hf_cache_dir=hf_cache_dir,
        max_examples=max_examples,
        answer_seed=answer_seed,
        exclude_subsets=exclude_subsets,
    )
    print(f"[LOAD_GPQA] loaded {len(rows)} rows  subsets={subsets}  exclude={exclude_subsets or 'none'}")
    if cache_normalized_path and rows:
        write_jsonl(cache_normalized_path, [r.to_dict() for r in rows])
        print(f"[LOAD_GPQA] cached normalized rows -> {cache_normalized_path}")
    elif cache_normalized_path:
        # Never cache an empty result (e.g. gated-dataset auth failure) — a
        # 0-row cache would be silently loaded by every subsequent run.
        print("[LOAD_GPQA] 0 rows loaded; NOT writing cache (fix HF auth and rerun).")
    return rows


# --------------------- Stage: MMLU-Pro loading ---------------------

def run_load_mmlu_pro(
    dataset_name: str = "TIGER-Lab/MMLU-Pro",
    categories: str = "",
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    splits: str = "test,validation",
    cache_normalized_path: Optional[str] = None,
) -> List[StandardRow]:
    split_list = [s.strip() for s in splits.split(",") if s.strip()]
    rows = load_mmlu_pro(
        dataset_name=dataset_name,
        categories=categories,
        hf_cache_dir=hf_cache_dir,
        max_examples=max_examples,
        splits=split_list,
    )
    cat_desc = categories or "all"
    print(f"[LOAD_MMLU_PRO] loaded {len(rows)} rows  categories={cat_desc}")
    if cache_normalized_path:
        write_jsonl(cache_normalized_path, [r.to_dict() for r in rows])
        print(f"[LOAD_MMLU_PRO] cached normalized rows -> {cache_normalized_path}")
    return rows


# --------------------- Stage: subagent SFT data synthesis ---------------------

def run_synthesize_subagent(
    ctx: StageContext,
    rows: List[StandardRow],
    agent_kind: AgentKind,
    teacher_provider: str,
    teacher_model: str,
    n_samples: int = 500,
    base_temperature: float = 0.4,
    max_retries: int = 2,
    use_cache: bool = True,
    max_workers: int = 8,
    symmetric_leakage: bool = False,
) -> Dict[str, Any]:
    from ..subagents.schemas import AgentKind
    from ..subagents.synthesize import synthesize_subagent_data

    agent_kind = AgentKind(_agent_kind_value(agent_kind))
    teacher = _build_teacher(teacher_provider, teacher_model, ctx)
    cache = TeacherCallCache(ctx.cache_dir) if use_cache else None
    auditor = LeakageAuditor()

    out_path = ctx.sft_jsonl_path(agent_kind.value)
    log_path = ctx.sft_log_path(agent_kind.value)

    stats = synthesize_subagent_data(
        rows=rows,
        agent_kind=agent_kind,
        teacher=teacher,
        out_path=out_path,
        cache=cache,
        auditor=auditor,
        n_samples=n_samples,
        base_temperature=base_temperature,
        max_retries_per_sample=max_retries,
        seed=ctx.seed,
        log_path=log_path,
        max_workers=max_workers,
        symmetric_leakage=symmetric_leakage,
    )

    return {
        "agent_kind": agent_kind.value,
        "teacher_provider": teacher.provider,
        "teacher_model": teacher.model,
        "out_path": out_path,
        "log_path": log_path,
        "stats": stats.__dict__,
    }


# --------------------- Stage: local DeepSeek JSONL bridge ---------------------

def run_export_deepseek_subagent_prompts(
    ctx: StageContext,
    rows: List[StandardRow],
    agent_kind: AgentKind,
    out_path: Optional[str] = None,
    n_samples: int = 500,
) -> Dict[str, Any]:
    """Write JSONL prompts for a local batch generator.

    Each row is compatible with the patched DeepSeek `generate_jsonl.py`:
      {"example_id": int, "prompt": [{"role": ..., "content": ...}, ...]}

    Extra fields are intentionally included so `import_deepseek_subagent_responses`
    can reconstruct validated SFT rows even if the generator output only keeps
    example_id/prompt/response.
    """
    from ..subagents.schemas import AgentKind
    from ..subagents.synthesize import _sample_verifier_candidate

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples] if n_samples > 0 else sample
    kind_value = _agent_kind_value(agent_kind)

    if out_path is None:
        out_path = os.path.join(ctx.sft_data_root, f"{kind_value}_deepseek_prompts.jsonl")

    out_rows: List[Dict[str, Any]] = []
    for r in sample:
        # Mirror the online-synthesis behavior: ~50% of verifier samples audit
        # a random candidate. The candidate is stored so the importer can build
        # the matching runtime prompt.
        candidate = (
            _sample_verifier_candidate(AgentKind.VERIFIER, r, ctx.seed)
            if kind_value == "verifier" else ""
        )
        prompt = _build_local_teacher_prompt(
            agent_kind,
            r,
            candidate_answer=candidate,
        )
        out_rows.append({
            "example_id": int(r.example_id),
            "question_hash": question_hash(r.question),
            "benchmark_name": r.benchmark_name,
            "agent_kind": kind_value,
            "question": r.question,
            "context": r.context,
            "choices": dict(r.choices),
            "ground_truth": r.ground_truth,
            "candidate_answer": candidate,
            "prompt": prompt,
        })

    write_jsonl(out_path, out_rows)
    return {"agent_kind": kind_value, "out_path": out_path, "n_rows": len(out_rows)}


def run_import_deepseek_subagent_responses(
    ctx: StageContext,
    agent_kind: AgentKind,
    prompt_jsonl: str,
    response_jsonl: str,
    out_path: Optional[str] = None,
    log_path: Optional[str] = None,
    teacher_model: str = "deepseek-local",
    raw_responses: bool = False,
) -> Dict[str, Any]:
    """Convert local JSONL responses into subagent SFT rows.

    By default responses are parsed, schema-validated, and leakage-audited. With
    raw_responses=True, keep the teacher response text exactly as generated but
    pair it with the runtime subagent prompt. This is useful for experiments
    that intentionally train on unfiltered teacher outputs without teaching the
    model the teacher-data-generation prompt.
    """
    from ..subagents.schemas import AgentKind
    from ..subagents.synthesize import (
        _extract_first_json,
        _gt_audit_keywords,
        _reasoner_choice_coverage_check,
        _validate_schema,
    )

    agent_kind = AgentKind(_agent_kind_value(agent_kind))
    kind_value = agent_kind.value
    if out_path is None:
        out_path = ctx.sft_jsonl_path(kind_value)
    if log_path is None:
        log_path = os.path.join(ctx.sft_data_root, f"{kind_value}_deepseek_import_log.jsonl")

    prompt_rows = read_jsonl(prompt_jsonl)
    response_rows = read_jsonl(response_jsonl)
    prompt_by_id = {int(r["example_id"]): r for r in prompt_rows if r.get("example_id") is not None}

    auditor = LeakageAuditor()
    sft_rows: List[Dict[str, Any]] = []
    log_rows: List[Dict[str, Any]] = []

    for resp_row in response_rows:
        eid = resp_row.get("example_id")
        try:
            eid_int = int(eid)
        except Exception:
            log_rows.append({"example_id": eid, "ok": False, "error": "missing_or_invalid_example_id"})
            continue

        src = prompt_by_id.get(eid_int)
        if src is None:
            log_rows.append({"example_id": eid_int, "ok": False, "error": "example_id_not_in_prompt_jsonl"})
            continue

        row = StandardRow(
            example_id=eid_int,
            benchmark_name=str(src.get("benchmark_name") or "medqa"),
            task_subtype=str(src.get("task_subtype") or ""),
            question=str(src.get("question") or ""),
            choices=dict(src.get("choices") or {}),
            ground_truth=str(src.get("ground_truth") or ""),
            context=str(src.get("context") or ""),
            metadata=dict(src.get("metadata") or {}),
            split=str(src.get("split") or ""),
        )

        text = str(resp_row.get("response") or "")
        # Use the SAME candidate the teacher prompt was built with (stored at
        # export time), so the runtime prompt matches the teacher's context.
        candidate = str(src.get("candidate_answer") or "")
        runtime_prompt = build_runtime_messages(
            agent_kind=kind_value,
            question=row.question,
            context=row.context,
            choices=row.choices,
            candidate_answer=candidate,
        )
        if raw_responses:
            if not text.strip():
                log_rows.append({"example_id": eid_int, "ok": False, "error": "empty_response"})
                continue
            sft_rows.append({
                "example_id": eid_int,
                "question_hash": question_hash(row.question),
                "benchmark_name": row.benchmark_name,
                "agent_kind": kind_value,
                "teacher_provider": "raw_jsonl",
                "teacher_model": teacher_model,
                "prompt": runtime_prompt,
                "response": text.strip(),
            })
            log_rows.append({"example_id": eid_int, "ok": True, "raw_response": True})
            continue

        obj = _extract_first_json(text)
        if obj is None:
            log_rows.append({
                "example_id": eid_int,
                "ok": False,
                "error": "json_parse_fail",
                "text_preview": text[:400],
            })
            continue

        try:
            model = _validate_schema(agent_kind, obj)
        except Exception as e:
            log_rows.append({
                "example_id": eid_int,
                "ok": False,
                "error": "schema_fail",
                "detail": str(e)[:400],
            })
            continue

        ok_balance, balance_msg = _reasoner_choice_coverage_check(agent_kind, obj, row)
        if not ok_balance:
            log_rows.append({
                "example_id": eid_int,
                "ok": False,
                "error": "balance_fail",
                "detail": balance_msg,
            })
            continue

        kw = _gt_audit_keywords(row)
        audit = auditor.audit(
            generated=obj,
            ground_truth_label=kw["ground_truth_label"],
            ground_truth_text=kw["ground_truth_text"],
            token_form=kw["token_form"],
        )
        if audit.leaked:
            log_rows.append({
                "example_id": eid_int,
                "ok": False,
                "error": "leakage_fail",
                "matches": audit.matches[:3],
            })
            continue

        sft_rows.append({
            "example_id": eid_int,
            "question_hash": question_hash(row.question),
            "benchmark_name": row.benchmark_name,
            "agent_kind": kind_value,
            "teacher_provider": "deepseek_local",
            "teacher_model": teacher_model,
            "candidate_answer": candidate,
            "prompt": runtime_prompt,
            "response": json.dumps(model.model_dump(), ensure_ascii=False),
        })
        log_rows.append({"example_id": eid_int, "ok": True})

    write_jsonl(out_path, sft_rows)
    write_jsonl(log_path, log_rows)
    return {
        "agent_kind": kind_value,
        "prompt_jsonl": prompt_jsonl,
        "response_jsonl": response_jsonl,
        "out_path": out_path,
        "log_path": log_path,
        "n_responses": len(response_rows),
        "n_imported": len(sft_rows),
        "n_failed": len(response_rows) - len(sft_rows),
        "raw_responses": raw_responses,
    }


# --------------------- Stage: subagent SFT training ---------------------

def run_train_subagent(
    ctx: StageContext,
    agent_kind: AgentKind,
    train_jsonl: Optional[str] = None,
    dev_jsonl: Optional[str] = None,
    epochs: int = 3,
    lr: float = 2e-4,
    max_seq_len: int = 4096,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_steps: int = -1,
) -> Dict[str, Any]:
    from ..subagents.train import SFTConfig, train_subagent_sft

    kind_value = _agent_kind_value(agent_kind)
    if train_jsonl is None:
        train_jsonl = ctx.sft_jsonl_path(kind_value)
    out_dir = ctx.adapter_path(kind_value)

    cfg = SFTConfig(
        base_model=ctx.base_model,
        train_jsonl=train_jsonl,
        dev_jsonl=dev_jsonl,
        out_dir=out_dir,
        max_seq_len=max_seq_len,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        seed=ctx.seed,
        max_steps=max_steps,
    )
    train_subagent_sft(cfg)
    return {"agent_kind": kind_value, "adapter_dir": out_dir, "train_jsonl": train_jsonl}


# --------------------- Stage: manager GRPO ---------------------

def run_train_manager_grpo(
    ctx: StageContext,
    train_rows: List[StandardRow],
    manager_adapter: Optional[str] = None,
    extractor_adapter: Optional[str] = None,
    reasoner_adapter: Optional[str] = None,
    verifier_adapter: Optional[str] = None,
    per_device_batch_size: int = 2,
    max_completion_length: int = 2048,
    temperature: float = 0.9,
    num_generations: int = 6,
    grpo_beta: float = 0.01,
    routing_efficiency_bonus: float = 0.0,
    tool_use_bonus: float = 0.0,
    ccr_mode: bool = False,
    ccr_p_high: float = 0.9,
    ccr_p_low: float = 0.2,
    ccr_k_max: int = 3,
    adc_mode: bool = False,
    adc_cost_per_tool: float = 0.05,
    adc_draft_bonus: float = 0.2,
    adc_missing_draft_penalty: float = 0.1,
    adc_final_bonus: float = 1.0,
    adc_variant: str = "anytime",
    full_parameter_rl: bool = False,
    max_steps: int = -1,
    output_dir: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "agent_routing",
    wandb_entity: str = "",
    wandb_run_name: str = "",
    task_description: str = "",
    subagent_server_url: Optional[str] = None,
    exploration_hint: str = "",
    clip_epsilon_high: float = 0.0,
) -> Dict[str, Any]:
    from ..manager.grpo_train import ManagerGRPOConfig, train_manager_grpo

    out_dir = output_dir or ctx.manager_grpo_dir()
    cfg = ManagerGRPOConfig(
        base_model=ctx.base_model,
        rows=train_rows,
        out_dir=out_dir,
        extractor_adapter=extractor_adapter or ctx.adapter_path("extractor"),
        reasoner_adapter=reasoner_adapter or ctx.adapter_path("reasoner"),
        verifier_adapter=verifier_adapter or ctx.adapter_path("verifier"),
        manager_adapter=manager_adapter,
        fail_buffer_jsonl=os.path.join(out_dir, "fail_buffer.jsonl"),
        raw_trace_jsonl=os.path.join(out_dir, "train_raw_trace.jsonl"),
        seed=ctx.seed,
        per_device_train_batch_size=per_device_batch_size,
        max_completion_length=max_completion_length,
        temperature=temperature,
        num_generations=num_generations,
        grpo_beta=grpo_beta,
        max_steps=max_steps,
        routing_efficiency_bonus=routing_efficiency_bonus,
        tool_use_bonus=tool_use_bonus,
        ccr_mode=ccr_mode,
        ccr_p_high=ccr_p_high,
        ccr_p_low=ccr_p_low,
        ccr_k_max=ccr_k_max,
        adc_mode=adc_mode,
        adc_cost_per_tool=adc_cost_per_tool,
        adc_draft_bonus=adc_draft_bonus,
        adc_missing_draft_penalty=adc_missing_draft_penalty,
        adc_final_bonus=adc_final_bonus,
        adc_variant=adc_variant,
        full_parameter_rl=full_parameter_rl,
        binding_mode=ctx.binding_mode,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        task_description=task_description,
        subagent_server_url=subagent_server_url,
        exploration_hint=exploration_hint,
        clip_epsilon_high=clip_epsilon_high,
    )
    train_manager_grpo(cfg)
    return {"manager_dir": out_dir, "fail_buffer": os.path.join(out_dir, "fail_buffer.jsonl")}


# --------------------- Stage: evolve build SFT ---------------------

def run_evolve_build_sft(
    ctx: StageContext,
    rows: List[StandardRow],
    teacher_provider: Optional[str] = None,
    teacher_model: Optional[str] = None,
    fail_buffer_jsonl: Optional[str] = None,
    max_fail_samples: int = 1500,
    task_description: str = "",
) -> Dict[str, Any]:
    from ..manager.evolve import EvolveSFTConfig, build_manager_sft_from_failures

    teacher = None
    if teacher_provider and teacher_model:
        teacher = _build_teacher(teacher_provider, teacher_model, ctx)

    fb = fail_buffer_jsonl or ctx.fail_buffer_path()
    out_dir = ctx.evolve_dir()
    cfg = EvolveSFTConfig(
        base_model=ctx.base_model,
        extractor_adapter=ctx.adapter_path("extractor"),
        reasoner_adapter=ctx.adapter_path("reasoner"),
        verifier_adapter=ctx.adapter_path("verifier"),
        rows=rows,
        fail_buffer_jsonl=fb,
        out_dir=out_dir,
        teacher=teacher,
        seed=ctx.seed,
        max_fail_samples=max_fail_samples,
        binding_mode=("argument" if ctx.binding_mode == "argument" else "environment"),
        task_description=task_description,
    )
    out_path = build_manager_sft_from_failures(cfg)
    return {"sft_jsonl": out_path, "out_dir": out_dir}


def run_export_manager_coldstart_prompts(
    ctx: StageContext,
    rows: List[StandardRow],
    n_samples: int = 300,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Export tool-sequence-selection prompts for offline batch generation.

    Output format mirrors the subagent deepseek export:
      {"example_id": int, "benchmark_name": str, "question": ..., "context": ...,
       "choices": {...}, "ground_truth": str, "binding_mode": str,
       "prompt": [{"role": "system", ...}, {"role": "user", ...}]}

    Feed the output to generate_openai_jsonl.py (or any local model) to get
    {"example_id": ..., "response": '{"tool_sequence": ["reasoner_tool"]}'} rows,
    then use run_import_manager_coldstart_responses to build SFT trajectories.
    """
    _AVAILABLE_TOOLS = ["extractor_tool", "reasoner_tool", "verifier_tool"]

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    if n_samples > 0:
        sample = sample[:n_samples]

    if out_path is None:
        os.makedirs(ctx.evolve_dir(), exist_ok=True)
        out_path = os.path.join(ctx.evolve_dir(), "coldstart_teacher_prompts.jsonl")

    sys_msg = (
        "You design efficient tool-use plans for a manager agent.\n"
        f"Available tools: {_AVAILABLE_TOOLS}.\n"
        "Choose a sequence of 0 to 3 tools (no repeats) that would best help a "
        "struggling manager solve the question.\n"
        "Return ONLY JSON: {\"tool_sequence\": [\"tool_a\", \"tool_b\"]}\n"
        "Use fewer tools when the question is simple."
    )

    out_rows: List[Dict[str, Any]] = []
    for r in sample:
        choices_block = ""
        if r.choices:
            lines = [f"  {k}. {v}" for k, v in r.choices.items()]
            choices_block = "CHOICES:\n" + "\n".join(lines) + "\n\n"
        user_msg = (
            f"QUESTION:\n{r.question}\n\n"
            f"{choices_block}"
            f"CONTEXT:\n{r.context if r.context else '(no context)'}\n"
        )
        binding = ctx.binding_mode if ctx.binding_mode in ("argument", "environment") else "environment"
        out_rows.append({
            "example_id": int(r.example_id),
            "question_hash": question_hash(r.question),
            "benchmark_name": r.benchmark_name,
            "question": r.question,
            "context": r.context,
            "choices": dict(r.choices),
            "ground_truth": r.ground_truth,
            "binding_mode": binding,
            "prompt": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        })

    write_jsonl(out_path, out_rows)
    print(f"[EXPORT_COLDSTART] {len(out_rows)} prompts -> {out_path}")
    return {"n_rows": len(out_rows), "out_path": out_path}


def run_import_manager_coldstart_responses(
    ctx: StageContext,
    prompt_jsonl: str,
    response_jsonl: str,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse teacher tool-sequence responses, run subagents, build SFT trajectories.

    Input response rows must have {"example_id": ..., "response": '{"tool_sequence": [...]}'}.
    Subagents are still run locally to produce tool outputs; only the tool *selection*
    decision comes from the offline-generated response.
    """
    from ..manager.evolve import ColdStartSFTConfig, build_manager_sft_from_sequences
    from ..benchmarks.base import StandardRow as _StandardRow

    prompt_rows = read_jsonl(prompt_jsonl)
    response_rows = read_jsonl(response_jsonl)
    prompt_by_id = {
        int(r["example_id"]): r for r in prompt_rows if r.get("example_id") is not None
    }

    _ALLOWED = {"extractor_tool", "reasoner_tool", "verifier_tool"}
    sequences: Dict[int, List[str]] = {}
    n_parse_fail = 0

    for resp_row in response_rows:
        eid = resp_row.get("example_id")
        try:
            eid_int = int(eid)
        except Exception:
            n_parse_fail += 1
            continue
        text = str(resp_row.get("response") or "")
        s = text.find("{")
        e_idx = text.rfind("}")
        seq: List[str] = []
        if s != -1 and e_idx > s:
            try:
                obj = json.loads(text[s : e_idx + 1])
                raw_seq = obj.get("tool_sequence", [])
                if isinstance(raw_seq, list):
                    seq = [t for t in raw_seq if t in _ALLOWED][:3]
                else:
                    n_parse_fail += 1
            except Exception:
                n_parse_fail += 1
        else:
            n_parse_fail += 1
        sequences[eid_int] = seq

    rows_for_build: List[StandardRow] = []
    for eid_int, src in prompt_by_id.items():
        if eid_int not in sequences:
            continue
        rows_for_build.append(_StandardRow(
            example_id=eid_int,
            benchmark_name=str(src.get("benchmark_name") or ""),
            task_subtype="",
            question=str(src.get("question") or ""),
            choices=dict(src.get("choices") or {}),
            ground_truth=str(src.get("ground_truth") or ""),
            context=str(src.get("context") or ""),
        ))

    binding = "environment"
    if prompt_rows:
        b = str(prompt_rows[0].get("binding_mode") or "environment")
        if b in ("argument", "environment"):
            binding = b

    if out_path is None:
        out_path = os.path.join(ctx.evolve_dir(), "coldstart_from_responses_sft.jsonl")

    cfg = ColdStartSFTConfig(
        base_model=ctx.base_model,
        extractor_adapter=ctx.adapter_path("extractor"),
        reasoner_adapter=ctx.adapter_path("reasoner"),
        verifier_adapter=ctx.adapter_path("verifier"),
        rows=rows_for_build,
        out_dir=ctx.evolve_dir(),
        teacher=None,
        seed=ctx.seed,
        n_samples=len(rows_for_build),
        binding_mode=binding,
        task_description="",
    )
    sft_path = build_manager_sft_from_sequences(cfg, sequences, out_path=out_path)

    print(
        f"[IMPORT_COLDSTART] prompts={len(prompt_rows)} responses={len(response_rows)} "
        f"parse_fail={n_parse_fail} examples={len(rows_for_build)}"
    )
    return {
        "prompt_jsonl": prompt_jsonl,
        "response_jsonl": response_jsonl,
        "sft_jsonl": sft_path,
        "n_examples": len(rows_for_build),
        "n_parse_fail": n_parse_fail,
    }


def run_manager_coldstart_sft(
    ctx: StageContext,
    rows: List[StandardRow],
    teacher_provider: Optional[str] = None,
    teacher_model: Optional[str] = None,
    n_samples: int = 300,
    task_description: str = "",
    epochs: int = 1,
    lr: float = 2e-5,
    max_seq_len: int = 4096,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    use_lora: bool = True,
    max_steps: int = -1,
    force_diverse: bool = False,
) -> Dict[str, Any]:
    from ..manager.evolve import (
        ColdStartSFTConfig, ManagerSFTConfig,
        build_manager_sft_from_rows, build_manager_sft_from_sequences,
        make_diverse_sequences, train_manager_sft,
    )

    data_dir = ctx.evolve_dir()
    binding = "argument" if ctx.binding_mode == "argument" else "environment"

    import random as _rng_mod
    sample = list(rows)
    _rng_mod.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    cfg = ColdStartSFTConfig(
        base_model=ctx.base_model,
        extractor_adapter=ctx.adapter_path("extractor"),
        reasoner_adapter=ctx.adapter_path("reasoner"),
        verifier_adapter=ctx.adapter_path("verifier"),
        rows=sample,
        out_dir=data_dir,
        teacher=None,
        seed=ctx.seed,
        n_samples=n_samples,
        binding_mode=binding,
        task_description=task_description,
    )

    if force_diverse:
        print("[COLDSTART] force_diverse=True: skipping teacher, using balanced sequence distribution")
        available_kinds = []
        for kind in ["extractor", "reasoner", "verifier"]:
            if os.path.isdir(ctx.adapter_path(kind)):
                available_kinds.append(kind)
        sequences = make_diverse_sequences(sample, available_kinds=available_kinds, seed=ctx.seed)
        sft_jsonl = os.path.join(data_dir, "manager_sft_coldstart_diverse.jsonl")
        sft_jsonl = build_manager_sft_from_sequences(cfg, sequences, out_path=sft_jsonl)
    else:
        teacher = None
        if teacher_provider and teacher_model:
            teacher = _build_teacher(teacher_provider, teacher_model, ctx)
        cfg.teacher = teacher
        sft_jsonl = build_manager_sft_from_rows(cfg)

    print(f"[COLDSTART] training manager on {sft_jsonl} ...")
    train_cfg = ManagerSFTConfig(
        base_model=ctx.base_model,
        train_jsonl=sft_jsonl,
        out_dir=ctx.manager_coldstart_dir(),
        seed=ctx.seed,
        max_seq_len=max_seq_len,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_lora=use_lora,
        max_steps=max_steps,
    )
    train_manager_sft(train_cfg)
    print(f"[COLDSTART] manager saved -> {ctx.manager_coldstart_dir()}")
    return {"sft_jsonl": sft_jsonl, "adapter_dir": ctx.manager_coldstart_dir()}


# --------------------- Stage: manager SFT (post-evolve) ---------------------

def run_train_manager_sft(
    ctx: StageContext,
    train_jsonl: Optional[str] = None,
    epochs: int = 1,
    lr: float = 2e-5,
    max_seq_len: int = 4096,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_steps: int = -1,
) -> Dict[str, Any]:
    from ..manager.evolve import ManagerSFTConfig, train_manager_sft

    if train_jsonl is None:
        train_jsonl = os.path.join(ctx.evolve_dir(), "manager_sft_from_failures.jsonl")
    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(f"manager SFT input not found: {train_jsonl}")

    out_dir = ctx.manager_sft_dir()
    cfg = ManagerSFTConfig(
        base_model=ctx.base_model,
        train_jsonl=train_jsonl,
        out_dir=out_dir,
        seed=ctx.seed,
        max_seq_len=max_seq_len,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_steps=max_steps,
    )
    train_manager_sft(cfg)
    return {"manager_sft_dir": out_dir}


# --------------------- Stage: full evolve round ---------------------

def run_evolve_round(
    ctx: StageContext,
    train_rows: List[StandardRow],
    full_rows: List[StandardRow],
    grpo_kwargs: Optional[Dict[str, Any]] = None,
    evolve_kwargs: Optional[Dict[str, Any]] = None,
    sft_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One full evolve round: GRPO -> build SFT from failures -> SFT manager."""
    grpo_kwargs = grpo_kwargs or {}
    evolve_kwargs = evolve_kwargs or {}
    sft_kwargs = sft_kwargs or {}

    grpo_res = run_train_manager_grpo(ctx=ctx, train_rows=train_rows, **grpo_kwargs)
    evolve_kwargs.setdefault("fail_buffer_jsonl", grpo_res.get("fail_buffer"))
    evolve_res = run_evolve_build_sft(ctx=ctx, rows=full_rows, **evolve_kwargs)
    sft_res = run_train_manager_sft(ctx=ctx, **sft_kwargs)
    return {"grpo": grpo_res, "evolve": evolve_res, "manager_sft": sft_res}


# --------------------- Stage: eval ---------------------

def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def run_eval_subagents(
    ctx: StageContext,
    rows: List[StandardRow],
    agent_kinds: List[AgentKind],
    n_samples: int = 50,
) -> Dict[str, Any]:
    """Evaluate each subagent's schema validity rate on a sample of rows.

    We do NOT score correctness here (subagents don't produce final answers);
    we score (1) does it return parseable JSON, (2) does it pass pydantic
    schema validation. This is the basic 'is the subagent functional' check.
    """
    import torch
    from ..subagents.runtime import FrozenSubagent, SubagentPool
    from ..subagents.schemas import AgentKind, SCHEMA_REGISTRY

    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    report: Dict[str, Any] = {
        "teacher_id": ctx.teacher_id, "n_samples": len(sample), "by_agent": {},
    }

    pool = SubagentPool()
    kinds = [AgentKind(_agent_kind_value(k)) for k in agent_kinds]
    for kind in kinds:
        adapter = ctx.adapter_path(kind.value)
        if not os.path.exists(adapter):
            print(f"[EVAL] adapter missing for {kind.value}: {adapter}; skipping.")
            continue
        pool.register(FrozenSubagent(ctx.base_model, adapter, kind.value, device))

    out_log_path = os.path.join(ctx.eval_root, "subagent_eval.jsonl")
    rows_log: List[Dict[str, Any]] = []

    for kind in kinds:
        if not pool.has(kind.value):
            continue
        n_total, n_json_ok, n_schema_ok = 0, 0, 0
        for r in sample:
            n_total += 1
            try:
                text = pool.call(
                    agent_kind=kind.value, example_id=r.example_id,
                    question=r.question, context=r.context, choices=r.choices,
                    cache_namespace=f"eval_{kind.value}",
                )
            except Exception as e:
                rows_log.append({"agent_kind": kind.value, "example_id": r.example_id,
                                 "error": str(e)[:300]})
                continue

            obj = _try_parse_json(text)
            if obj is None:
                rows_log.append({"agent_kind": kind.value, "example_id": r.example_id,
                                 "json_ok": False, "schema_ok": False,
                                 "raw_preview": text[:300]})
                continue
            n_json_ok += 1

            schema_cls = SCHEMA_REGISTRY[kind]
            try:
                schema_cls(**obj)
                n_schema_ok += 1
                rows_log.append({"agent_kind": kind.value, "example_id": r.example_id,
                                 "json_ok": True, "schema_ok": True})
            except Exception as e:
                rows_log.append({"agent_kind": kind.value, "example_id": r.example_id,
                                 "json_ok": True, "schema_ok": False,
                                 "schema_error": str(e)[:300]})

        report["by_agent"][kind.value] = {
            "n_total": n_total,
            "json_ok_rate": (n_json_ok / n_total) if n_total else 0.0,
            "schema_ok_rate": (n_schema_ok / n_total) if n_total else 0.0,
        }

    write_jsonl(out_log_path, rows_log)
    write_json(os.path.join(ctx.eval_root, "subagent_eval_report.json"), report)
    print("[EVAL/SUBAGENT]", report["by_agent"])
    return report


def run_eval_manager(
    ctx: StageContext,
    rows: List[StandardRow],
    manager_dir: Optional[str] = None,
    n_samples: int = 100,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    task_description: str = "",
    sc_k: int = 1,
    sc_temperature: float = 0.7,
) -> Dict[str, Any]:
    """Evaluate manager accuracy + routing pattern on a sample of rows.

    Note: this uses a SIMPLE one-shot generation (no native tool calling).
    For tool-using eval you'd need to set up the same TRL rollout machinery
    as training; this is a pragmatic accuracy probe.

    sc_k > 1 enables a self-consistency baseline: sample sc_k completions at
    sc_temperature and take the majority vote over parsed answers. This is the
    matched-compute resampling control for RQ1 (compare its token budget to the
    learned orchestrator's delegation budget).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if manager_dir is None:
        manager_dir = (
            ctx.manager_sft_dir() if os.path.exists(ctx.manager_sft_dir()) else ctx.manager_grpo_dir()
        )
    if not os.path.exists(manager_dir):
        raise FileNotFoundError(f"manager_dir not found: {manager_dir}")

    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    tok = AutoTokenizer.from_pretrained(manager_dir, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    is_full = (
        os.path.exists(os.path.join(manager_dir, "config.json"))
        and not os.path.exists(os.path.join(manager_dir, "adapter_config.json"))
    )
    if is_full:
        model = AutoModelForCausalLM.from_pretrained(
            manager_dir, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            ctx.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base, manager_dir).to(device)
    model.eval()

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    rows_log: List[Dict[str, Any]] = []
    n_correct = 0
    _iter = _tqdm(sample, desc="eval_manager", unit="ex") if _tqdm else sample
    for r in _iter:
        sys_prompt = build_manager_system_prompt(
            label_keys=list(r.choices.keys()), task_description=task_description,
        )
        user_msg = build_manager_user_message(
            example_id=r.example_id, question=r.question,
            context=r.context, choices=r.choices, binding_mode="argument",
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            prompt_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = tok(prompt_text, return_tensors="pt").to(device)

        if sc_k > 1:
            votes: List[str] = []
            previews: List[str] = []
            for _ in range(sc_k):
                gen = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=max(sc_temperature, 1e-6),
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
                out = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                previews.append(out[:200])
                p = parse_final_answer(out, list(r.choices.keys()))
                if p is not None:
                    votes.append(p)
            if votes:
                from collections import Counter
                pred = Counter(votes).most_common(1)[0][0]
            else:
                pred = None
            out = " ||| ".join(previews)
        else:
            do_sample = temperature > 1e-6
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                **({"temperature": max(temperature, 1e-6)} if do_sample else {}),
            )
            out = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            pred = parse_final_answer(out, list(r.choices.keys()))

        correct = bool(pred is not None and pred == r.ground_truth)
        if correct:
            n_correct += 1
        rows_log.append({
            "example_id": r.example_id, "ground_truth": r.ground_truth,
            "pred": pred, "correct": correct, "output_preview": out[:600],
        })
        done = len(rows_log)
        if _tqdm and hasattr(_iter, "set_postfix"):
            _iter.set_postfix(acc=f"{n_correct/done:.3f}", correct=n_correct, n=done)

    accuracy = n_correct / max(1, len(sample))
    suffix = f"_sc{sc_k}" if sc_k > 1 else ""
    report = {
        "teacher_id": ctx.teacher_id, "manager_dir": manager_dir,
        "n_samples": len(sample), "accuracy": accuracy,
        "sc_k": sc_k,
    }
    write_jsonl(os.path.join(ctx.eval_root, f"manager_eval{suffix}.jsonl"), rows_log)
    write_json(os.path.join(ctx.eval_root, f"manager_eval_report{suffix}.json"), report)
    print(f"[EVAL/MANAGER] teacher={ctx.teacher_id} acc={accuracy:.3f} sc_k={sc_k} (n={len(sample)})")
    return report


def _manager_tool_schemas(binding_mode: str) -> List[Dict[str, Any]]:
    required = ["example_id"] if binding_mode == "argument" else []
    properties = (
        {
            "example_id": {
                "type": "integer",
                "description": "The current example ID from the user message.",
            }
        }
        if binding_mode == "argument"
        else {}
    )
    verifier_properties = dict(properties)
    verifier_properties["current_draft"] = {
        "type": "string",
        "description": "Your current draft answer key (e.g. \"B\") to audit.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "extractor_tool",
                "description": "Extract decision-relevant factual signals from the question and context.",
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reasoner_tool",
                "description": "Produce a structured reasoning scaffold for the choices.",
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verifier_tool",
                "description": "Identify relevant domain principles and audit the reasoning for logical or computational errors. Pass your current draft answer via current_draft.",
                "parameters": {"type": "object", "properties": verifier_properties, "required": required},
            },
        },
    ]


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _extract_manager_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse Qwen-style XML tool calls emitted by the chat template."""
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        name = str(obj.get("name") or "").strip()
        args = obj.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name:
            calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    content = _TOOL_CALL_RE.sub("", text or "").strip()
    return content, calls


def _tool_call_message(
    tool_name: str, args: Dict[str, Any], call_id: str, content: str = ""
) -> Dict[str, Any]:
    return {
        "role": "assistant",
        # Keep the assistant's own text (DRAFT_ANSWER_ etc.) in the history so
        # eval matches training, where TRL preserves tool-call turn content.
        "content": content,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }],
    }


def _load_manager_for_eval(ctx: StageContext, manager_dir: str, device: str, dtype: Any):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(manager_dir, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    is_full = (
        os.path.exists(os.path.join(manager_dir, "config.json"))
        and not os.path.exists(os.path.join(manager_dir, "adapter_config.json"))
    )
    if is_full:
        model = AutoModelForCausalLM.from_pretrained(
            manager_dir, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    else:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            ctx.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model = PeftModel.from_pretrained(base, manager_dir).to(device)
    model.eval()
    return tok, model


def _render_manager_chat(tok: Any, messages: List[Dict[str, Any]],
                         tools: List[Dict[str, Any]]) -> str:
    try:
        return tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )


def run_eval_manager_tools(
    ctx: StageContext,
    rows: List[StandardRow],
    manager_dir: Optional[str] = None,
    n_samples: int = 100,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    max_tool_calls: int = 3,
    task_description: str = "",
    subagent_server_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the manager with the same frozen subagents used as tools."""
    import torch
    from ..subagents.runtime import FrozenSubagent, SubagentPool

    if manager_dir is None:
        manager_dir = ctx.manager_grpo_dir()
    if not os.path.exists(manager_dir):
        raise FileNotFoundError(f"manager_dir not found: {manager_dir}")

    binding_mode = ctx.binding_mode
    if binding_mode == "auto":
        run_config = os.path.join(manager_dir, "manager_run_config.json")
        if os.path.exists(run_config):
            try:
                with open(run_config, "r", encoding="utf-8") as f:
                    binding_mode = str(json.load(f).get("binding_mode") or "argument")
            except Exception:
                binding_mode = "argument"
        else:
            binding_mode = "argument"
    if binding_mode == "environment":
        # The local XML tool loop is equivalent to argument binding except the
        # example ID is injected by the evaluator instead of generated by model.
        user_binding_mode = "environment"
    else:
        user_binding_mode = "argument"

    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if subagent_server_url:
        from ..subagents.runtime import RemoteSubagentPool
        pool = RemoteSubagentPool(server_url=subagent_server_url)
        print(f"[EVAL] using remote subagent pool -> {subagent_server_url}")
    else:
        pool = SubagentPool()
        # subagent_base_model = "Qwen/Qwen3-4B"
        subagent_base_model = ctx.base_model
        for kind in ("extractor", "reasoner", "verifier"):
            adapter = ctx.adapter_path(kind)
            if os.path.exists(adapter):
                pool.register(FrozenSubagent(subagent_base_model, adapter, kind, device))
        if not pool._agents:
            raise FileNotFoundError(f"No subagent adapters found under {ctx.adapter_root}")

    # pool = SubagentPool()
    # for kind in ("extractor", "reasoner", "verifier"):
    #     adapter = ctx.adapter_path(kind)
    #     if os.path.exists(adapter):
    #         pool.register(FrozenSubagent(ctx.base_model, adapter, kind, device))
    # if not pool._agents:
    #     raise FileNotFoundError(f"No subagent adapters found under {ctx.adapter_root}")

    tok, model = _load_manager_for_eval(ctx, manager_dir, device, dtype)
    tools = _manager_tool_schemas(user_binding_mode)

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    try:
        from tqdm import tqdm as _tqdm2
    except ImportError:
        _tqdm2 = None

    rows_log: List[Dict[str, Any]] = []
    n_correct = 0
    n_valid = 0
    total_tool_calls = 0
    tool_counts: Dict[str, int] = {}
    malformed_tool_calls = 0

    _iter2 = _tqdm2(sample, desc="eval_manager_tools", unit="ex") if _tqdm2 else sample
    for r in _iter2:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_manager_system_prompt(
                    label_keys=list(r.choices.keys()),
                    task_description=task_description,
                ),
            },
            {
                "role": "user",
                "content": build_manager_user_message(
                    example_id=r.example_id,
                    question=r.question,
                    context=r.context,
                    choices=r.choices,
                    binding_mode=user_binding_mode,
                ),
            },
        ]
        trajectory: List[Dict[str, Any]] = []
        used_tools: List[str] = []
        final_text = ""

        for step in range(max(1, max_tool_calls + 1)):
            prompt_text = _render_manager_chat(tok, messages, tools)
            inputs = tok(prompt_text, return_tensors="pt").to(device)
            do_sample = temperature > 1e-6
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                **({"temperature": max(temperature, 1e-6)} if do_sample else {}),
            )
            out = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            content, calls = _extract_manager_tool_calls(out)
            final_text = content or out

            if not calls or len(used_tools) >= max_tool_calls:
                messages.append({"role": "assistant", "content": final_text})
                trajectory.append({"role": "assistant", "content": final_text[:2000], "tool_calls": []})
                break

            call = calls[0]
            tool_name = call["name"]
            args = dict(call.get("arguments") or {})
            if user_binding_mode == "environment" or "example_id" not in args:
                args["example_id"] = int(r.example_id)

            call_id = f"eval_{int(r.example_id)}_{len(used_tools)}"
            messages.append(_tool_call_message(tool_name, args, call_id, content=content))
            used_tools.append(tool_name)
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

            tool_kind = tool_name[:-5] if tool_name.endswith("_tool") else tool_name
            candidate = str(args.get("current_draft") or "") if tool_kind == "verifier" else ""
            try:
                tool_output = pool.call(
                    agent_kind=tool_kind,
                    example_id=int(args.get("example_id", r.example_id)),
                    question=r.question,
                    context=r.context,
                    choices=r.choices,
                    cache_namespace="eval_manager_tools",
                    candidate_answer=candidate,
                )
            except Exception as e:
                malformed_tool_calls += 1
                tool_output = json.dumps({"error": str(e)}, ensure_ascii=False)

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": tool_output,
            })
            trajectory.append({
                "role": "assistant",
                "content": content[:1000],
                "tool_call": {"name": tool_name, "arguments": args},
            })
            trajectory.append({
                "role": "tool",
                "name": tool_name,
                "content": tool_output[:2000],
            })

        pred = parse_final_answer(final_text, list(r.choices.keys()))
        correct = bool(pred is not None and pred == r.ground_truth)
        if pred is not None:
            n_valid += 1
        if correct:
            n_correct += 1
        total_tool_calls += len(used_tools)
        done2 = len(rows_log) + 1
        if _tqdm2 and hasattr(_iter2, "set_postfix"):
            _iter2.set_postfix(
                acc=f"{n_correct/done2:.3f}",
                tools=f"{total_tool_calls/done2:.1f}",
                n=done2,
            )
        rows_log.append({
            "example_id": r.example_id,
            "benchmark_name": r.benchmark_name,
            "task_subtype": r.task_subtype,
            "ground_truth": r.ground_truth,
            "pred": pred,
            "correct": correct,
            "valid_answer": pred is not None,
            "tool_calls": len(used_tools),
            "tool_names_called": used_tools,
            "final_text": final_text[:2000],
            "trajectory": trajectory,
        })

    n = len(sample)
    report = {
        "teacher_id": ctx.teacher_id,
        "manager_dir": manager_dir,
        "n_samples": n,
        "accuracy": n_correct / max(1, n),
        "valid_answer_rate": n_valid / max(1, n),
        "tool_call_rate": sum(1 for r in rows_log if r["tool_calls"] > 0) / max(1, n),
        "avg_tool_calls": total_tool_calls / max(1, n),
        "tool_counts": tool_counts,
        "malformed_tool_calls": malformed_tool_calls,
        "binding_mode": binding_mode,
        # "subagents": sorted(pool._agents.keys()),
      "subagents": sorted(pool._agents.keys()) if hasattr(pool, '_agents') else ["remote"],
    }
    write_jsonl(os.path.join(ctx.eval_root, "manager_tool_eval.jsonl"), rows_log)
    write_json(os.path.join(ctx.eval_root, "manager_tool_eval_report.json"), report)
    print(
        f"[EVAL/MANAGER_TOOLS] teacher={ctx.teacher_id} "
        f"acc={report['accuracy']:.3f} tool_rate={report['tool_call_rate']:.3f} (n={n})"
    )
    return report


def run_eval_manager_forced(
    ctx: StageContext,
    rows: List[StandardRow],
    manager_dir: Optional[str] = None,
    forced_tools: Optional[List[str]] = None,
    n_samples: int = 100,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    task_description: str = "",
    out_tag: str = "",
    subagent_server_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the manager under a FIXED delegation sequence (no free choice).

    For each forced advisor, the assistant tool-call turn and the frozen
    advisor's output are injected into the history (mirroring cold-start SFT
    construction); the manager generates only the final answer turn.

    Running this once per advisor subset yields (a) the fixed-k baselines for
    the RQ1 main table and the RQ2 Pareto plot, and (b) the per-question
    inputs for the stopping oracle: oracle reward = max over subsets of
    (correct - cost * k), computed offline from the saved jsonl files.

    The verifier runs its generic audit here (no candidate is passed — the
    manager has not stated a draft in forced mode, and passing ground truth
    would leak).
    """
    import torch
    from ..subagents.runtime import FrozenSubagent, SubagentPool

    forced = [t.strip() for t in (forced_tools or []) if t.strip() and t.strip() != "none"]
    valid_kinds = {"extractor", "reasoner", "verifier"}
    for t in forced:
        if t not in valid_kinds:
            raise ValueError(f"forced tool must be one of {sorted(valid_kinds)}, got {t!r}")

    if manager_dir is None:
        manager_dir = ctx.manager_grpo_dir()
    if not os.path.exists(manager_dir):
        raise FileNotFoundError(f"manager_dir not found: {manager_dir}")

    binding_mode = ctx.binding_mode
    if binding_mode == "auto":
        run_config = os.path.join(manager_dir, "manager_run_config.json")
        if os.path.exists(run_config):
            try:
                with open(run_config, "r", encoding="utf-8") as f:
                    binding_mode = str(json.load(f).get("binding_mode") or "argument")
            except Exception:
                binding_mode = "argument"
        else:
            binding_mode = "argument"

    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if subagent_server_url:
        from ..subagents.runtime import RemoteSubagentPool
        pool = RemoteSubagentPool(server_url=subagent_server_url)
        print(f"[EVAL] using remote subagent pool -> {subagent_server_url}")
    else:
        pool = SubagentPool()
        subagent_base_model = getattr(ctx, "subagent_base_model", "") or ctx.base_model
        for kind in ("extractor", "reasoner", "verifier"):
            adapter = ctx.adapter_path(kind)
            if os.path.exists(adapter):
                pool.register(FrozenSubagent(subagent_base_model, adapter, kind, device))
        if not pool._agents:
            raise FileNotFoundError(f"No subagent adapters found under {ctx.adapter_root}")
        # Fail fast: forced eval often runs many subsets back-to-back;
        # a missing adapter should die here, not at the first pool.call.
        for t in forced:
            if not pool.has(t):
                raise FileNotFoundError(
                    f"forced tool {t!r} has no adapter under {ctx.adapter_root}"
                )
    # pool = SubagentPool()
    # for kind in ("extractor", "reasoner", "verifier"):
    #     adapter = ctx.adapter_path(kind)
    #     if os.path.exists(adapter):
    #         pool.register(FrozenSubagent(ctx.base_model, adapter, kind, device))
    # for t in forced:
    #     if not pool.has(t):
    #         raise FileNotFoundError(f"forced tool {t} has no adapter under {ctx.adapter_root}")

    tok, model = _load_manager_for_eval(ctx, manager_dir, device, dtype)
    tools = _manager_tool_schemas(binding_mode if binding_mode == "argument" else "environment")

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    try:
        from tqdm import tqdm as _tqdm3
    except ImportError:
        _tqdm3 = None

    rows_log: List[Dict[str, Any]] = []
    n_correct = 0
    n_valid = 0
    _iter3 = _tqdm3(sample, desc=f"eval_forced[{','.join(forced) or 'none'}]", unit="ex") if _tqdm3 else sample
    for r in _iter3:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_manager_system_prompt(
                    label_keys=list(r.choices.keys()),
                    task_description=task_description,
                ),
            },
            {
                "role": "user",
                "content": build_manager_user_message(
                    example_id=r.example_id,
                    question=r.question,
                    context=r.context,
                    choices=r.choices,
                    binding_mode=binding_mode,
                ),
            },
        ]
        for i, kind in enumerate(forced):
            tool_name = f"{kind}_tool"
            args: Dict[str, Any] = (
                {"example_id": int(r.example_id)} if binding_mode == "argument" else {}
            )
            call_id = f"forced_{int(r.example_id)}_{i}"
            messages.append(_tool_call_message(tool_name, args, call_id))
            tool_output = pool.call(
                agent_kind=kind,
                example_id=int(r.example_id),
                question=r.question,
                context=r.context,
                choices=r.choices,
                cache_namespace="eval_forced",
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": tool_output,
            })

        prompt_text = _render_manager_chat(tok, messages, tools)
        inputs = tok(prompt_text, return_tensors="pt").to(device)
        do_sample = temperature > 1e-6
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            **({"temperature": max(temperature, 1e-6)} if do_sample else {}),
        )
        out = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        content, _extra_calls = _extract_manager_tool_calls(out)
        final_text = content or out
        pred = parse_final_answer(final_text, list(r.choices.keys()))
        correct = bool(pred is not None and pred == r.ground_truth)
        if pred is not None:
            n_valid += 1
        if correct:
            n_correct += 1
        rows_log.append({
            "example_id": r.example_id,
            "benchmark_name": r.benchmark_name,
            "task_subtype": r.task_subtype,
            "ground_truth": r.ground_truth,
            "pred": pred,
            "correct": correct,
            "valid_answer": pred is not None,
            "tool_calls": len(forced),
            "forced_tools": list(forced),
            "final_text": final_text[:1200],
        })

    n = len(sample)
    tag = out_tag or (",".join(forced) if forced else "none")
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    report = {
        "teacher_id": ctx.teacher_id,
        "manager_dir": manager_dir,
        "forced_tools": list(forced),
        "k": len(forced),
        "n_samples": n,
        "accuracy": n_correct / max(1, n),
        "valid_answer_rate": n_valid / max(1, n),
        "binding_mode": binding_mode,
    }
    write_jsonl(os.path.join(ctx.eval_root, f"manager_forced_{safe_tag}.jsonl"), rows_log)
    write_json(os.path.join(ctx.eval_root, f"manager_forced_{safe_tag}_report.json"), report)
    print(
        f"[EVAL/FORCED] tools=[{tag}] acc={report['accuracy']:.3f} "
        f"valid={report['valid_answer_rate']:.3f} (n={n})"
    )
    return report


# --------------------- Stage: harness stopping (counterfactuals / probe) ---------------------
#
# These stages implement marginal-utility-optimal stopping WITHOUT touching
# the RL algorithm or the reward function:
#   collect_counterfactuals -> fit_stop_probe -> eval_manager_adaptive
#   (optional) export_stop_distill_sft -> train_manager_sft
# See src/manager/stopping.py for the rationale and the pure-logic pieces.

_CANONICAL_TOOL_ORDER = ("extractor", "reasoner", "verifier")


def _resolve_binding_mode(ctx: StageContext, manager_dir: str) -> str:
    """Same resolution rule as run_eval_manager_tools / run_eval_manager_forced."""
    binding_mode = ctx.binding_mode
    if binding_mode == "auto":
        run_config = os.path.join(manager_dir, "manager_run_config.json")
        if os.path.exists(run_config):
            try:
                with open(run_config, "r", encoding="utf-8") as f:
                    binding_mode = str(json.load(f).get("binding_mode") or "argument")
            except Exception:
                binding_mode = "argument"
        else:
            binding_mode = "argument"
    return binding_mode if binding_mode in ("argument", "environment") else "argument"


def _build_stop_subagent_pool(
    ctx: StageContext,
    subagent_server_url: Optional[str],
    device: str,
) -> Tuple[Any, List[str]]:
    """Build a (pool, available_kinds) pair; works for local and remote pools."""
    from ..subagents.runtime import FrozenSubagent, RemoteSubagentPool, SubagentPool

    if subagent_server_url:
        pool: Any = RemoteSubagentPool(server_url=subagent_server_url)
        print(f"[STOP] using remote subagent pool -> {subagent_server_url}")
    else:
        pool = SubagentPool()
        subagent_base_model = getattr(ctx, "subagent_base_model", "") or ctx.base_model
        for kind in _CANONICAL_TOOL_ORDER:
            adapter = ctx.adapter_path(kind)
            if os.path.exists(adapter):
                pool.register(FrozenSubagent(subagent_base_model, adapter, kind, device))
    available = [k for k in _CANONICAL_TOOL_ORDER if pool.has(k)]
    if not available:
        raise FileNotFoundError(f"No subagents available under {ctx.adapter_root}")
    return pool, available


def _generate_manager_texts(
    model: Any,
    tok: Any,
    device: str,
    prompt_text: str,
    n: int,
    temperature: float,
    max_new_tokens: int,
) -> List[str]:
    """Generate n completions for one rendered prompt.

    Greedy decoding requires n == 1; sampled decoding uses a single batched
    generate call via num_return_sequences.
    """
    inputs = tok(prompt_text, return_tensors="pt").to(device)
    plen = inputs["input_ids"].shape[1]
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if temperature > 1e-6:
        gen_kwargs.update(
            do_sample=True,
            temperature=max(temperature, 1e-6),
            num_return_sequences=max(1, int(n)),
        )
    else:
        if n != 1:
            raise ValueError("Greedy decoding supports exactly n=1 completion.")
        gen_kwargs.update(do_sample=False)
    out = model.generate(**inputs, **gen_kwargs)
    return [
        tok.decode(seq[plen:], skip_special_tokens=True).strip() for seq in out
    ]


def _parse_probe_pred(text: str, choice_keys: List[str]) -> Tuple[Optional[str], str]:
    """Parse a probe completion into an answer key.

    Priority: strict final ANSWER_ line, then last DRAFT_ANSWER_ (models
    occasionally emit only the draft form under the forced-answer nudge).
    """
    pred = parse_final_answer(text, choice_keys)
    if pred is not None:
        return pred, "final"
    pred = parse_draft_answer(text, choice_keys)
    if pred is not None:
        return pred, "draft"
    return None, "none"


def _stage_answer_probe(
    model: Any,
    tok: Any,
    device: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    choice_keys: List[str],
    n_votes: int,
    vote_temperature: float,
    probe_max_new_tokens: int,
) -> Dict[str, Any]:
    """Branch off the trajectory and force an answer at the current stage.

    The probe user message is appended on a COPY of the history; the main
    trajectory is never contaminated. Returns the greedy answer (used as the
    stage prediction and, on stopping, as the final answer), the greedy text,
    and n_votes sampled answers for the self-consistency confidence features.
    """
    probe_messages = messages + [{"role": "user", "content": stoplib.STOP_PROBE_USER_MSG}]
    prompt_text = _render_manager_chat(tok, probe_messages, tools)

    greedy_out = _generate_manager_texts(
        model, tok, device, prompt_text, n=1, temperature=0.0,
        max_new_tokens=probe_max_new_tokens,
    )[0]
    content, _ = _extract_manager_tool_calls(greedy_out)
    greedy_text = content or greedy_out
    pred, pred_source = _parse_probe_pred(greedy_text, choice_keys)

    votes: List[Optional[str]] = []
    if n_votes > 0:
        vote_outs = _generate_manager_texts(
            model, tok, device, prompt_text, n=n_votes,
            temperature=vote_temperature, max_new_tokens=probe_max_new_tokens,
        )
        for v_out in vote_outs:
            v_content, _ = _extract_manager_tool_calls(v_out)
            v_pred, _src = _parse_probe_pred(v_content or v_out, choice_keys)
            votes.append(v_pred)

    return {
        "pred": pred,
        "pred_source": pred_source,
        "final_text": greedy_text,
        "votes": votes,
        "n_generations": 1 + max(0, n_votes),
    }


def _normalize_tool_choice(
    calls: List[Dict[str, Any]],
    used_kinds: List[str],
    available_kinds: List[str],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Map the model's first tool call to a valid unused kind, or None."""
    for call in calls:
        name = str(call.get("name") or "").strip()
        kind = name[:-5] if name.endswith("_tool") else name
        if kind in available_kinds and kind not in used_kinds:
            return kind, dict(call.get("arguments") or {})
    return None, {}


def _stop_tool_args(
    binding_mode: str,
    example_id: int,
    kind: str,
    stage_pred: Optional[str],
    model_args: Optional[Dict[str, Any]],
    choice_keys: List[str],
) -> Dict[str, Any]:
    """Canonical tool-call arguments for the recorded/executed history turn."""
    args: Dict[str, Any] = (
        {"example_id": int(example_id)} if binding_mode == "argument" else {}
    )
    if kind == "verifier":
        draft = ""
        model_draft = str((model_args or {}).get("current_draft") or "").strip()
        if model_draft in choice_keys:
            draft = model_draft
        elif stage_pred is not None:
            draft = str(stage_pred)
        if draft:
            args["current_draft"] = draft
    return args


def run_collect_counterfactuals(
    ctx: StageContext,
    rows: List[StandardRow],
    manager_dir: Optional[str] = None,
    n_samples: int = 0,
    k_max: int = 3,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    n_votes: int = 5,
    vote_temperature: float = 0.7,
    probe_max_new_tokens: int = 512,
    task_description: str = "",
    subagent_server_url: Optional[str] = None,
    out_tag: str = "cf",
) -> Dict[str, Any]:
    """Forced-continuation rollouts with a per-stage answer probe.

    For every question the manager is driven through k_max tool calls
    (its own choice when it makes a valid unused call; otherwise the next
    unused tool in canonical order is injected), and at EVERY stage
    k = 0..k_max a probe branch forces an answer (greedy) plus n_votes
    sampled answers. The written JSONL contains, per question, the full
    marginal-benefit curve and everything needed to (a) fit the stop probe,
    (b) sweep stopping thresholds offline, and (c) export distillation SFT.
    """
    import torch

    if manager_dir is None:
        manager_dir = ctx.manager_grpo_dir()
    if not os.path.exists(manager_dir):
        raise FileNotFoundError(f"manager_dir not found: {manager_dir}")
    if k_max < 0:
        raise ValueError(f"k_max must be >= 0, got {k_max}")

    binding_mode = _resolve_binding_mode(ctx, manager_dir)
    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pool, available_kinds = _build_stop_subagent_pool(ctx, subagent_server_url, device)
    tok, model = _load_manager_for_eval(ctx, manager_dir, device, dtype)
    tools = _manager_tool_schemas(binding_mode)

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    if n_samples > 0:
        sample = sample[:n_samples]

    try:
        from tqdm import tqdm as _tqdm_cf
    except ImportError:
        _tqdm_cf = None

    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", out_tag or "cf")
    out_jsonl = os.path.join(ctx.eval_root, f"counterfactual_{safe_tag}.jsonl")
    # Start fresh: appending across runs would duplicate example_ids.
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    records: List[Dict[str, Any]] = []
    n_probe_generations = 0
    n_model_chosen = 0
    n_injected = 0
    n_model_wanted_stop = 0

    _iter = _tqdm_cf(sample, desc=f"collect_cf[{safe_tag}]", unit="ex") if _tqdm_cf else sample
    for r in _iter:
        choice_keys = list(r.choices.keys())
        base_messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_manager_system_prompt(
                    label_keys=choice_keys,
                    task_description=task_description,
                ),
            },
            {
                "role": "user",
                "content": build_manager_user_message(
                    example_id=r.example_id,
                    question=r.question,
                    context=r.context,
                    choices=r.choices,
                    binding_mode=binding_mode,
                ),
            },
        ]
        messages = [dict(m) for m in base_messages]
        used_kinds: List[str] = []
        stages_rec: List[Dict[str, Any]] = []
        steps_rec: List[Dict[str, Any]] = []

        for k in range(k_max + 1):
            probe = _stage_answer_probe(
                model, tok, device, messages, tools, choice_keys,
                n_votes=n_votes, vote_temperature=vote_temperature,
                probe_max_new_tokens=probe_max_new_tokens,
            )
            n_probe_generations += probe["n_generations"]
            vs = stoplib.vote_stats(probe["votes"], len(choice_keys))
            stages_rec.append({
                "k": k,
                "pred": probe["pred"],
                "pred_source": probe["pred_source"],
                "correct": bool(
                    probe["pred"] is not None and probe["pred"] == r.ground_truth
                ),
                "votes": probe["votes"],
                "vote_majority": vs["majority"],
                "vote_agreement": round(vs["agreement_top"], 4),
                "valid_vote_frac": round(vs["valid_vote_frac"], 4),
                "vote_entropy_norm": round(vs["vote_entropy_norm"], 4),
                "final_text": probe["final_text"],
            })
            if k == k_max:
                break
            unused = [t for t in available_kinds if t not in used_kinds]
            if not unused:
                break

            # Continuation turn: let the manager pick the next tool; inject the
            # next unused canonical tool when it stops early or calls invalidly.
            prompt_text = _render_manager_chat(tok, messages, tools)
            cont_out = _generate_manager_texts(
                model, tok, device, prompt_text, n=1,
                temperature=temperature, max_new_tokens=max_new_tokens,
            )[0]
            content, calls = _extract_manager_tool_calls(cont_out)
            kind, model_args = _normalize_tool_choice(calls, used_kinds, available_kinds)
            model_wanted_stop = not calls

            if kind is not None:
                chosen_by = "model"
                n_model_chosen += 1
                turn_content = stoplib.answer_to_draft(content)
            else:
                kind = unused[0]
                chosen_by = "injected"
                n_injected += 1
                if model_wanted_stop:
                    n_model_wanted_stop += 1
                stage_pred = stages_rec[-1]["pred"]
                turn_content = (
                    f"DRAFT_ANSWER_{stoplib._label_to_token(str(stage_pred))}"
                    if stage_pred is not None else ""
                )
                model_args = {}

            tool_name = f"{kind}_tool"
            args = _stop_tool_args(
                binding_mode, int(r.example_id), kind,
                stages_rec[-1]["pred"], model_args, choice_keys,
            )
            call_id = f"cf_{int(r.example_id)}_{len(used_kinds)}"
            messages.append(_tool_call_message(tool_name, args, call_id, content=turn_content))
            try:
                tool_output = pool.call(
                    agent_kind=kind,
                    example_id=int(r.example_id),
                    question=r.question,
                    context=r.context,
                    choices=r.choices,
                    cache_namespace=f"cf_{safe_tag}",
                    candidate_answer=str(args.get("current_draft") or ""),
                )
            except Exception as e:
                tool_output = json.dumps({"error": str(e)}, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": tool_output,
            })
            used_kinds.append(kind)
            steps_rec.append({
                "i": len(used_kinds) - 1,
                "tool_name": tool_name,
                "kind": kind,
                "chosen_by": chosen_by,
                "model_wanted_stop": bool(model_wanted_stop),
                "assistant_content": turn_content,
                "args": args,
                "call_id": call_id,
                "tool_output": tool_output,
            })

        record = {
            "example_id": int(r.example_id),
            "question_hash": question_hash(r.question),
            "benchmark_name": r.benchmark_name,
            "task_subtype": r.task_subtype,
            "ground_truth": r.ground_truth,
            "choice_keys": choice_keys,
            "n_choices": len(choice_keys),
            "question_len": len(r.question or ""),
            "context_len": len(r.context or ""),
            "k_max": k_max,
            "binding_mode": binding_mode,
            "base_messages": base_messages,
            "stages": stages_rec,
            "steps": steps_rec,
        }
        records.append(record)
        append_jsonl(out_jsonl, [record])
        if _tqdm_cf and hasattr(_iter, "set_postfix"):
            accs = [
                sum(1 for rec in records if stoplib._stage_correct(rec, min(kk, len(rec["stages"]) - 1)))
                / max(1, len(records))
                for kk in (0, k_max)
            ]
            _iter.set_postfix(acc0=f"{accs[0]:.3f}", accK=f"{accs[1]:.3f}", n=len(records))

    n = len(records)
    report = {
        "teacher_id": ctx.teacher_id,
        "manager_dir": manager_dir,
        "n_samples": n,
        "k_max": k_max,
        "n_votes": n_votes,
        "vote_temperature": vote_temperature,
        "binding_mode": binding_mode,
        "fixed_k_accuracy": stoplib.fixed_k_stats(records, k_max=k_max),
        "oracle": stoplib.oracle_stats(records),
        "heuristics": stoplib.heuristic_stats(records),
        "marginal_tool_table": stoplib.marginal_tool_table(records),
        "tool_choice": {
            "model_chosen_steps": n_model_chosen,
            "injected_steps": n_injected,
            "model_wanted_stop_steps": n_model_wanted_stop,
        },
        "avg_probe_generations_per_example": round(n_probe_generations / max(1, n), 2),
        "out_jsonl": out_jsonl,
    }
    write_json(os.path.join(ctx.eval_root, f"counterfactual_{safe_tag}_report.json"), report)
    print(
        f"[COLLECT_CF] tag={safe_tag} n={n} "
        f"fixed_k={ {kk: v['accuracy'] for kk, v in report['fixed_k_accuracy'].items()} } "
        f"oracle={report['oracle']}"
    )
    return report


def run_fit_stop_probe(
    ctx: StageContext,
    train_jsonls: List[str],
    eval_jsonl: Optional[str] = None,
    l2: float = 1e-3,
    epsilon: float = 0.005,
    holdout_frac: float = 0.25,
    out_path: Optional[str] = None,
    out_tag: str = "",
) -> Dict[str, Any]:
    """Fit the stop probe and choose the deployment threshold.

    Threshold selection NEVER sees the questions used for weight fitting:
    either pass a separate --probe_eval_jsonl, or a deterministic
    per-question holdout split is carved from the training records.
    """
    records: List[Dict[str, Any]] = []
    for path in train_jsonls:
        if not os.path.exists(path):
            raise FileNotFoundError(f"counterfactual jsonl not found: {path}")
        records.extend(read_jsonl(path))
    if not records:
        raise ValueError("No counterfactual records loaded.")

    if eval_jsonl:
        if not os.path.exists(eval_jsonl):
            raise FileNotFoundError(f"probe eval jsonl not found: {eval_jsonl}")
        fit_records = records
        sel_records = read_jsonl(eval_jsonl)
    else:
        fit_records, sel_records = stoplib.stable_holdout_split(records, holdout_frac)
    if len(fit_records) < 5 or len(sel_records) < 5:
        raise ValueError(
            f"Too few records after split: fit={len(fit_records)} "
            f"select={len(sel_records)}. Collect more counterfactuals."
        )

    probe, fit_metrics = stoplib.fit_stop_probe(fit_records, l2=l2)

    # Holdout probe quality (same metrics, unseen questions).
    hold_rows = [row for rec in sel_records for row in stoplib.record_stage_feature_rows(rec)]
    hold_p = [probe.predict_proba(row["features"]) for row in hold_rows]
    hold_y = [row["label"] for row in hold_rows]
    hold_metrics = {
        "n_stage_rows": len(hold_rows),
        "auc": stoplib.rank_auc(hold_y, hold_p),
        "brier": round(
            sum((p - y) ** 2 for p, y in zip(hold_p, hold_y)) / max(1, len(hold_rows)), 4
        ),
        "ece": stoplib.expected_calibration_error(hold_y, hold_p),
    }

    k_max = max((int(rec.get("k_max") or stoplib.K_MAX_DEFAULT) for rec in sel_records), default=3)
    sweep = stoplib.threshold_sweep(sel_records, probe)
    fixed = stoplib.fixed_k_stats(sel_records, k_max=k_max)
    acc_full = fixed[str(k_max)]["accuracy"]
    choice = stoplib.choose_threshold(sweep, acc_full, epsilon)

    probe.threshold = float(choice["threshold"])
    probe.meta.update({
        "teacher_id": ctx.teacher_id,
        "train_jsonls": list(train_jsonls),
        "eval_jsonl": eval_jsonl or f"holdout_{holdout_frac}",
        "threshold_choice": choice,
    })

    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", out_tag) if out_tag else ""
    suffix = f"_{safe_tag}" if safe_tag else ""
    probe_path = out_path or os.path.join(ctx.eval_root, f"stop_probe{suffix}.json")
    probe.save(probe_path)

    report = {
        "teacher_id": ctx.teacher_id,
        "probe_path": probe_path,
        "n_fit_records": len(fit_records),
        "n_select_records": len(sel_records),
        "fit_metrics": fit_metrics,
        "holdout_metrics": hold_metrics,
        "threshold_choice": choice,
        "fixed_k_accuracy": fixed,
        "oracle": stoplib.oracle_stats(sel_records),
        "heuristics": stoplib.heuristic_stats(sel_records),
        "marginal_tool_table": stoplib.marginal_tool_table(sel_records),
        "sweep": sweep,
    }
    write_json(os.path.join(ctx.eval_root, f"stop_probe{suffix}_report.json"), report)
    print(
        f"[FIT_STOP_PROBE] probe={probe_path} "
        f"holdout_auc={hold_metrics['auc']} choice={choice}"
    )
    return report


def run_eval_manager_adaptive(
    ctx: StageContext,
    rows: List[StandardRow],
    probe_path: str,
    manager_dir: Optional[str] = None,
    stop_threshold: float = -1.0,
    n_samples: int = 100,
    k_max: int = 3,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    n_votes: int = 5,
    vote_temperature: float = 0.7,
    probe_max_new_tokens: int = 512,
    force_continue: bool = False,
    task_description: str = "",
    subagent_server_url: Optional[str] = None,
    out_tag: str = "",
) -> Dict[str, Any]:
    """Deployment eval: the manager runs its normal free-choice loop, but the
    HARNESS terminates the episode at the first stage where the stop probe is
    confident (P >= threshold). The stage answer probe doubles as the final
    answer generator on stopping, so a probe-stopped episode costs exactly the
    probe generations plus the tools actually executed.

    force_continue=True additionally overrides the manager's own early stops
    (injecting the next unused tool) while the probe is unconfident — the
    harness then fully owns the stopping decision in both directions.
    """
    import torch

    if manager_dir is None:
        manager_dir = ctx.manager_grpo_dir()
    if not os.path.exists(manager_dir):
        raise FileNotFoundError(f"manager_dir not found: {manager_dir}")

    probe = stoplib.StopProbe.load(probe_path)
    threshold = float(stop_threshold) if stop_threshold >= 0 else float(probe.threshold)

    binding_mode = _resolve_binding_mode(ctx, manager_dir)
    set_seed(ctx.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pool, available_kinds = _build_stop_subagent_pool(ctx, subagent_server_url, device)
    tok, model = _load_manager_for_eval(ctx, manager_dir, device, dtype)
    tools = _manager_tool_schemas(binding_mode)

    sample = list(rows)
    random.Random(ctx.seed).shuffle(sample)
    sample = sample[:n_samples]

    try:
        from tqdm import tqdm as _tqdm_ad
    except ImportError:
        _tqdm_ad = None

    rows_log: List[Dict[str, Any]] = []
    n_correct = 0
    n_valid = 0
    total_tool_calls = 0
    total_probe_generations = 0
    stop_reasons: Dict[str, int] = {}

    _iter = _tqdm_ad(sample, desc=f"eval_adaptive[thr={threshold:.2f}]", unit="ex") if _tqdm_ad else sample
    for r in _iter:
        choice_keys = list(r.choices.keys())
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": build_manager_system_prompt(
                    label_keys=choice_keys, task_description=task_description,
                ),
            },
            {
                "role": "user",
                "content": build_manager_user_message(
                    example_id=r.example_id, question=r.question,
                    context=r.context, choices=r.choices,
                    binding_mode=binding_mode,
                ),
            },
        ]
        used_kinds: List[str] = []
        prev_pred: Optional[str] = None
        pred: Optional[str] = None
        stop_reason = "budget_exhausted"
        probe_generations = 0
        stage_probs: List[float] = []

        for k in range(k_max + 1):
            probe_res = _stage_answer_probe(
                model, tok, device, messages, tools, choice_keys,
                n_votes=n_votes, vote_temperature=vote_temperature,
                probe_max_new_tokens=probe_max_new_tokens,
            )
            probe_generations += probe_res["n_generations"]
            feats = stoplib.stage_features(
                k=k, k_max=k_max, greedy_pred=probe_res["pred"],
                votes=probe_res["votes"], prev_pred=prev_pred,
                n_choices=len(choice_keys),
                question_len=len(r.question or ""),
                context_len=len(r.context or ""),
            )
            p_stop = probe.predict_proba(feats)
            stage_probs.append(round(p_stop, 4))
            unused = [t for t in available_kinds if t not in used_kinds]

            if p_stop >= threshold:
                pred = probe_res["pred"]
                stop_reason = "probe_confident"
                break
            if k == k_max:
                pred = probe_res["pred"]
                stop_reason = "budget_exhausted"
                break
            if not unused:
                pred = probe_res["pred"]
                stop_reason = "no_tools_left"
                break

            prompt_text = _render_manager_chat(tok, messages, tools)
            cont_out = _generate_manager_texts(
                model, tok, device, prompt_text, n=1,
                temperature=temperature, max_new_tokens=max_new_tokens,
            )[0]
            content, calls = _extract_manager_tool_calls(cont_out)
            kind, model_args = _normalize_tool_choice(calls, used_kinds, available_kinds)

            if kind is None:
                if force_continue:
                    kind = unused[0]
                    model_args = {}
                    turn_content = (
                        f"DRAFT_ANSWER_{stoplib._label_to_token(str(probe_res['pred']))}"
                        if probe_res["pred"] is not None else ""
                    )
                else:
                    own = parse_final_answer(content or cont_out, choice_keys)
                    if own is not None:
                        pred = own
                        stop_reason = "model_stopped"
                    else:
                        pred = probe_res["pred"]
                        stop_reason = "model_stopped_unparsed"
                    break
            else:
                turn_content = stoplib.answer_to_draft(content)

            tool_name = f"{kind}_tool"
            args = _stop_tool_args(
                binding_mode, int(r.example_id), kind,
                probe_res["pred"], model_args, choice_keys,
            )
            call_id = f"ad_{int(r.example_id)}_{len(used_kinds)}"
            messages.append(_tool_call_message(tool_name, args, call_id, content=turn_content))
            try:
                tool_output = pool.call(
                    agent_kind=kind,
                    example_id=int(r.example_id),
                    question=r.question,
                    context=r.context,
                    choices=r.choices,
                    cache_namespace="eval_adaptive",
                    candidate_answer=str(args.get("current_draft") or ""),
                )
            except Exception as e:
                tool_output = json.dumps({"error": str(e)}, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": tool_output,
            })
            used_kinds.append(kind)
            prev_pred = probe_res["pred"]

        correct = bool(pred is not None and pred == r.ground_truth)
        if pred is not None:
            n_valid += 1
        if correct:
            n_correct += 1
        total_tool_calls += len(used_kinds)
        total_probe_generations += probe_generations
        stop_reasons[stop_reason] = stop_reasons.get(stop_reason, 0) + 1
        rows_log.append({
            "example_id": r.example_id,
            "benchmark_name": r.benchmark_name,
            "task_subtype": r.task_subtype,
            "ground_truth": r.ground_truth,
            "pred": pred,
            "correct": correct,
            "valid_answer": pred is not None,
            "tool_calls": len(used_kinds),
            "tool_names_called": [f"{k}_tool" for k in used_kinds],
            "stop_reason": stop_reason,
            "stage_probs": stage_probs,
            "probe_generations": probe_generations,
        })
        done = len(rows_log)
        if _tqdm_ad and hasattr(_iter, "set_postfix"):
            _iter.set_postfix(
                acc=f"{n_correct/done:.3f}",
                tools=f"{total_tool_calls/done:.2f}",
                n=done,
            )

    n = len(sample)
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", out_tag) if out_tag else ""
    suffix = f"_{safe_tag}" if safe_tag else ""
    report = {
        "teacher_id": ctx.teacher_id,
        "manager_dir": manager_dir,
        "probe_path": probe_path,
        "threshold": threshold,
        "force_continue": bool(force_continue),
        "n_samples": n,
        "accuracy": n_correct / max(1, n),
        "valid_answer_rate": n_valid / max(1, n),
        "avg_tool_calls": total_tool_calls / max(1, n),
        "tool_call_rate": sum(1 for x in rows_log if x["tool_calls"] > 0) / max(1, n),
        "avg_probe_generations": total_probe_generations / max(1, n),
        "n_votes": n_votes,
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "binding_mode": binding_mode,
    }
    write_jsonl(os.path.join(ctx.eval_root, f"manager_adaptive_eval{suffix}.jsonl"), rows_log)
    write_json(os.path.join(ctx.eval_root, f"manager_adaptive_eval{suffix}_report.json"), report)
    print(
        f"[EVAL/ADAPTIVE] thr={threshold:.2f} acc={report['accuracy']:.3f} "
        f"avg_tools={report['avg_tool_calls']:.2f} reasons={report['stop_reasons']} (n={n})"
    )
    return report


def run_export_stop_distill_sft(
    ctx: StageContext,
    cf_jsonl: str,
    probe_path: str,
    stop_threshold: float = -1.0,
    out_path: Optional[str] = None,
    require_correct: bool = True,
    max_examples: int = 0,
) -> Dict[str, Any]:
    """Convert probe-selected minimal correct trajectories into manager SFT rows.

    Feed the output to train_manager_sft (same per-turn prompt/response format
    as manager/evolve.py). This distills the harness stopping rule back into
    the policy without touching GRPO or the reward function.
    """
    if not os.path.exists(cf_jsonl):
        raise FileNotFoundError(f"counterfactual jsonl not found: {cf_jsonl}")
    probe = stoplib.StopProbe.load(probe_path)
    threshold = float(stop_threshold) if stop_threshold >= 0 else float(probe.threshold)

    records = read_jsonl(cf_jsonl)
    if max_examples > 0:
        records = records[:max_examples]

    sft_rows: List[Dict[str, Any]] = []
    k_star_dist: Dict[str, int] = {}
    n_used = 0
    n_skipped = 0
    for rec in records:
        k_star = stoplib.pick_stop_index(rec, probe, threshold)
        rows = stoplib.build_distill_rows(rec, k_star, require_correct=require_correct)
        if not rows:
            n_skipped += 1
            continue
        n_used += 1
        k_star_dist[str(k_star)] = k_star_dist.get(str(k_star), 0) + 1
        sft_rows.extend(rows)

    if out_path is None:
        os.makedirs(ctx.evolve_dir(), exist_ok=True)
        out_path = os.path.join(ctx.evolve_dir(), "stop_distill_sft.jsonl")
    write_jsonl(out_path, sft_rows)

    report = {
        "teacher_id": ctx.teacher_id,
        "cf_jsonl": cf_jsonl,
        "probe_path": probe_path,
        "threshold": threshold,
        "require_correct": bool(require_correct),
        "n_records": len(records),
        "n_examples_used": n_used,
        "n_examples_skipped": n_skipped,
        "k_star_dist": dict(sorted(k_star_dist.items())),
        "n_sft_rows": len(sft_rows),
        "out_path": out_path,
    }
    write_json(out_path + ".report.json", report)
    print(
        f"[EXPORT_STOP_DISTILL] examples={n_used}/{len(records)} "
        f"k_star={report['k_star_dist']} rows={len(sft_rows)} -> {out_path}"
    )
    return report
