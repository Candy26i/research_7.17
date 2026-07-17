"""Build GPQA caches with the complete Diamond split excluded from training.

GPQA subsets are NESTED: diamond (198) ⊆ main (448) ⊆ extended (546).
This script creates:

  - eval : all 198 Diamond questions (or a reporting subset)
  - train: Extended minus *all* Diamond questions (348 questions)

Even when --eval_n is smaller than 198, unreported Diamond questions remain
excluded from training so a later full-Diamond evaluation stays valid.

Requires: accepted dataset terms on HF + `huggingface-cli login`.

Usage (from agent_routing/):
    python scripts/build_gpqa_splits.py                # full 198 eval
    python scripts/build_gpqa_splits.py --eval_n 198 --seed 42 --out_dir outputs/data
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks.base import question_hash
from src.benchmarks.gpqa import load_gpqa
from src.utils.io import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_n", type=int, default=0,
                    help="Number of Diamond questions to write; 0 means all 198. "
                         "All Diamond questions remain excluded from training.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/data")
    ap.add_argument("--answer_seed", type=int, default=42)
    args = ap.parse_args()

    ext = load_gpqa(subsets="gpqa_extended", answer_seed=args.answer_seed)
    dia = load_gpqa(subsets="gpqa_diamond", answer_seed=args.answer_seed)
    if not ext or not dia:
        sys.exit("Failed to load GPQA (gated dataset — accept terms + huggingface-cli login).")

    dia_hashes = {question_hash(r.question) for r in dia}
    non_dia = [r for r in ext if question_hash(r.question) not in dia_hashes]
    matched = sum(question_hash(r.question) in dia_hashes for r in ext)
    print(f"extended={len(ext)}  diamond={len(dia)}  matched_in_ext={matched}  non_diamond={len(non_dia)}")
    if matched < len(dia):
        sys.exit("Diamond/Extended hash matching is incomplete; refusing to build splits.")

    rng = random.Random(args.seed)
    eval_rows = list(dia)
    rng.shuffle(eval_rows)
    if args.eval_n > 0:
        eval_rows = eval_rows[: min(args.eval_n, len(eval_rows))]
    train_rows = list(non_dia)
    rng.shuffle(train_rows)

    for r in eval_rows:
        r.split = "test"
    for r in train_rows:
        r.split = "train"

    # Contamination guard: eval and train must be hash-disjoint.
    eval_h = {question_hash(r.question) for r in eval_rows}
    all_diamond_h = {question_hash(r.question) for r in dia}
    train_h = {question_hash(r.question) for r in train_rows}
    assert not (eval_h & train_h), "eval/train overlap detected"
    assert not (all_diamond_h & train_h), "Diamond contamination detected in train"

    os.makedirs(args.out_dir, exist_ok=True)
    eval_path = os.path.join(args.out_dir, f"gpqa_diamond_eval{len(eval_rows)}.jsonl")
    train_path = os.path.join(args.out_dir, f"gpqa_nondiamond_train{len(train_rows)}.jsonl")
    write_jsonl(eval_path, [r.to_dict() for r in eval_rows])
    write_jsonl(train_path, [r.to_dict() for r in train_rows])
    print(f"eval  -> {eval_path}  ({len(eval_rows)} rows, all diamond, split=test)")
    print(f"train -> {train_path}  ({len(train_rows)} non-diamond rows, split=train)")


if __name__ == "__main__":
    main()
