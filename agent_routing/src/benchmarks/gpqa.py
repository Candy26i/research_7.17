"""GPQA loader.

Loads the Idavidrein/gpqa dataset from HuggingFace.

GPQA has three configurations:
  - gpqa_main     (448 questions)
  - gpqa_diamond  (198 questions — hardest subset, recommended for eval)
  - gpqa_extended (546 questions)

NOTE: Idavidrein/gpqa is a gated dataset. You must accept the terms on
HuggingFace (https://huggingface.co/datasets/Idavidrein/gpqa) and run
`huggingface-cli login` before loading.

Each question has one correct answer and three incorrect answers. We
shuffle them into A/B/C/D using a per-example seeded RNG so the mapping
is deterministic but not trivially always "A".
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import StandardRow

HF_DEFAULT_DATASET = "Idavidrein/gpqa"
VALID_SUBSETS = ("gpqa_main", "gpqa_diamond", "gpqa_extended")
DEFAULT_SUBSET = "gpqa_diamond"


def _shuffle_answers(
    correct: str,
    incorrects: List[str],
    seed: int,
) -> Tuple[Dict[str, str], str]:
    """Shuffle correct + incorrect answers into A/B/C/D and return ground truth key."""
    answers = [correct] + incorrects
    rng = random.Random(seed)
    rng.shuffle(answers)
    choices = {chr(ord("A") + i): ans for i, ans in enumerate(answers)}
    gt = next(k for k, v in choices.items() if v == correct)
    return choices, gt


def _from_record(
    rec: Dict[str, Any],
    idx: int,
    subset: str,
    answer_seed: int,
) -> Optional[StandardRow]:
    question = str(rec.get("Question") or "").strip()
    correct = str(rec.get("Correct Answer") or "").strip()
    if not question or not correct:
        return None

    incorrects = []
    for i in range(1, 4):
        val = str(rec.get(f"Incorrect Answer {i}") or "").strip()
        if val:
            incorrects.append(val)

    if len(incorrects) < 3:
        return None

    # Per-example seed = global seed XOR idx, keeps shuffle deterministic
    choices, gt = _shuffle_answers(correct, incorrects, seed=answer_seed ^ idx)

    subdomain = str(rec.get("Subdomain") or rec.get("subdomain") or "").strip()
    domain = str(
        rec.get("High-level domain")
        or rec.get("Domain")
        or rec.get("domain")
        or ""
    ).strip()

    return StandardRow(
        example_id=idx,
        benchmark_name="gpqa",
        task_subtype=subset,
        question=question,
        choices=choices,
        ground_truth=gt,
        context="",
        metadata={"subdomain": subdomain, "domain": domain, "subset": subset},
        split="test",
    )


def _normalize_question(q: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", str(q).strip().lower())


def _collect_subset_questions(
    dataset_name: str,
    subsets: List[str],
    hf_cache_dir: Optional[str],
) -> set:
    """Collect normalized question texts of the given subsets (for exclusion)."""
    from datasets import load_dataset

    questions: set = set()
    for subset in subsets:
        try:
            ds = load_dataset(dataset_name, subset, cache_dir=hf_cache_dir)
        except Exception as e:
            print(f"[LOAD_GPQA] WARNING: could not load exclusion subset '{subset}': {e}")
            continue
        for split_name in ds.keys():
            for rec in ds[split_name]:
                q = str(dict(rec).get("Question") or "").strip()
                if q:
                    questions.add(_normalize_question(q))
    return questions


def load_gpqa(
    dataset_name: str = HF_DEFAULT_DATASET,
    subsets: "Sequence[str] | str" = DEFAULT_SUBSET,
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    answer_seed: int = 42,
    exclude_subsets: "Sequence[str] | str" = "",
) -> List[StandardRow]:
    """Load GPQA into a list of StandardRow.

    Args:
        dataset_name: HuggingFace dataset id (must be accepted/gated).
        subsets: one of gpqa_main/gpqa_diamond/gpqa_extended, a comma
                 string of those names, or "all".
        hf_cache_dir: optional HuggingFace cache directory.
        max_examples: cap total examples; 0 means no cap.
        answer_seed: seed for answer shuffling to ensure deterministic A/B/C/D.
        exclude_subsets: subsets whose questions are removed from the result.
            GPQA subsets are NESTED (diamond ⊆ main ⊆ extended), so training on
            main/extended while evaluating on diamond REQUIRES
            exclude_subsets="gpqa_diamond" to avoid contamination.
    """
    from datasets import load_dataset

    if isinstance(subsets, str):
        s = subsets.strip()
        if s.lower() == "all":
            subset_list = list(VALID_SUBSETS)
        else:
            subset_list = [x.strip() for x in s.split(",") if x.strip()]
    else:
        subset_list = [str(s).strip() for s in subsets if str(s).strip()]

    if not subset_list:
        subset_list = [DEFAULT_SUBSET]

    if isinstance(exclude_subsets, str):
        exclude_list = [x.strip() for x in exclude_subsets.split(",") if x.strip()]
    else:
        exclude_list = [str(x).strip() for x in exclude_subsets if str(x).strip()]
    excluded_questions = (
        _collect_subset_questions(dataset_name, exclude_list, hf_cache_dir)
        if exclude_list else set()
    )

    rows: List[StandardRow] = []
    n_excluded = 0
    for subset in subset_list:
        try:
            ds = load_dataset(dataset_name, subset, cache_dir=hf_cache_dir)
        except Exception as e:
            print(
                f"[LOAD_GPQA] WARNING: could not load subset '{subset}' from "
                f"'{dataset_name}'. "
                f"Make sure you accepted the dataset terms on HuggingFace and "
                f"are logged in (`huggingface-cli login`). Error: {e}"
            )
            continue

        for split_name in ds.keys():
            for rec in ds[split_name]:
                rec_d = dict(rec)
                if excluded_questions:
                    q_norm = _normalize_question(str(rec_d.get("Question") or ""))
                    if q_norm in excluded_questions:
                        n_excluded += 1
                        continue
                sr = _from_record(rec_d, len(rows), subset, answer_seed)
                if sr is not None:
                    rows.append(sr)
                if max_examples > 0 and len(rows) >= max_examples:
                    break
            if max_examples > 0 and len(rows) >= max_examples:
                break
        if max_examples > 0 and len(rows) >= max_examples:
            break

    if exclude_list:
        print(f"[LOAD_GPQA] excluded {n_excluded} rows overlapping {exclude_list}")

    # Reassign contiguous example_ids
    for new_id, r in enumerate(rows):
        r.example_id = new_id

    return rows
