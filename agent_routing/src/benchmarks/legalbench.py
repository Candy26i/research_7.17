"""LegalBench loader.

Loads HuggingFace `nguha/legalbench` configs and converts classification-like
tasks into StandardRow. LegalBench contains many heterogeneous tasks; this
loader intentionally keeps the first pass conservative:

  - expects rows with `text` and `answer`
  - builds choices from the answer label set for each config
  - skips configs with too many labels, which are usually not suitable for
    the current multiple-choice manager/subagent pipeline
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import StandardRow


def _label_key(i: int) -> str:
    if i < 26:
        return chr(ord("A") + i)
    return f"L{i + 1}"


def _normalize_answer(raw: Any) -> str:
    return "" if raw is None else str(raw).strip()


def _row_text(rec: Dict[str, Any]) -> str:
    text = rec.get("text")
    if text is not None:
        return str(text).strip()
    # Fallback for odd configs: concatenate non-answer fields.
    parts = []
    for k, v in rec.items():
        if k in {"answer", "label", "index", "idx"}:
            continue
        if v is not None:
            parts.append(f"{k}: {v}")
    return "\n".join(parts).strip()


def _build_question(config_name: str, text: str) -> str:
    return (
        f"LegalBench task: {config_name}\n\n"
        f"TEXT:\n{text}\n\n"
        "Classify this text into the most appropriate label from the choices."
    )


def _get_config_names(dataset_name: str) -> List[str]:
    from datasets import get_dataset_config_names

    return list(get_dataset_config_names(dataset_name))


def _load_config_splits(
    dataset_name: str,
    config_name: str,
    cache_dir: Optional[str],
) -> Dict[str, List[Dict[str, Any]]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in ds.keys():
        norm = "dev" if str(split_name).lower() == "validation" else str(split_name).lower()
        out[norm] = [dict(r) for r in ds[split_name]]
    return out


def _label_space(split_rows: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    labels = set()
    for rows in split_rows.values():
        for rec in rows:
            ans = _normalize_answer(rec.get("answer", rec.get("label")))
            if ans:
                labels.add(ans)
    return sorted(labels)


def load_legalbench(
    dataset_name: str = "nguha/legalbench",
    configs: Sequence[str] | str = (),
    split: str = "test",
    cache_dir: Optional[str] = None,
    max_examples: int = 0,
    max_labels: int = 12,
) -> Tuple[List[StandardRow], Dict[str, Any]]:
    """Load LegalBench configs into StandardRow.

    Args:
        dataset_name: HuggingFace dataset id.
        configs: config names, comma string, or "all".
        split: split to draw examples from, usually "test" because LegalBench
            train splits are often only few-shot demonstrations.
        cache_dir: optional HF cache dir.
        max_examples: cap total loaded rows; 0 means no cap.
        max_labels: skip configs whose answer label space is larger than this.
    """
    if isinstance(configs, str):
        configs_s = configs.strip()
        if configs_s.lower() == "all":
            config_names = _get_config_names(dataset_name)
        else:
            config_names = [c.strip() for c in configs_s.split(",") if c.strip()]
    else:
        config_names = [str(c).strip() for c in configs if str(c).strip()]

    if not config_names:
        raise ValueError("LegalBench requires --legalbench_configs, e.g. 'abercrombie' or 'all'.")

    target_split = split.lower().strip() or "test"
    rows: List[StandardRow] = []
    skipped: List[Dict[str, Any]] = []

    for config_name in config_names:
        try:
            split_rows = _load_config_splits(dataset_name, config_name, cache_dir)
        except Exception as e:
            skipped.append({"config": config_name, "reason": f"load_error: {e}"})
            continue

        labels = _label_space(split_rows)
        if len(labels) < 2:
            skipped.append({"config": config_name, "reason": f"label_count={len(labels)}"})
            continue
        if len(labels) > max_labels:
            skipped.append({"config": config_name, "reason": f"label_count={len(labels)} > {max_labels}"})
            continue

        split_name = target_split
        records = split_rows.get(split_name)
        if records is None and split_name == "dev":
            records = split_rows.get("validation")
        if records is None:
            skipped.append({"config": config_name, "reason": f"missing_split={target_split}"})
            continue

        choices = {_label_key(i): label for i, label in enumerate(labels)}
        key_by_label = {v: k for k, v in choices.items()}

        for rec in records:
            answer = _normalize_answer(rec.get("answer", rec.get("label")))
            gt = key_by_label.get(answer)
            text = _row_text(rec)
            if not gt or not text:
                continue
            rows.append(StandardRow(
                example_id=len(rows),
                benchmark_name="legalbench",
                task_subtype=config_name,
                question=_build_question(config_name, text),
                choices=choices,
                ground_truth=gt,
                context="",
                metadata={
                    "legalbench_config": config_name,
                    "source_index": rec.get("index", rec.get("idx")),
                    "answer_text": answer,
                },
                split=target_split,
            ))
            if max_examples > 0 and len(rows) >= max_examples:
                meta = {
                    "dataset_name": dataset_name,
                    "configs_requested": config_names,
                    "split": target_split,
                    "max_labels": max_labels,
                    "skipped": skipped,
                }
                return rows, meta

    meta = {
        "dataset_name": dataset_name,
        "configs_requested": config_names,
        "split": target_split,
        "max_labels": max_labels,
        "skipped": skipped,
    }
    return rows, meta
