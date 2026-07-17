"""Build GPQA train/eval caches with a diamond-held-out eval set.

GPQA subsets are NESTED: diamond (198) ⊆ main (448) ⊆ extended (546).
This script splits the 546-question extended set into:

  - eval : N diamond questions (seeded sample)      -> gpqa_diamond_eval{N}.jsonl
  - train: remaining diamond + all non-diamond      -> gpqa_train{546-N}.jsonl

Disjointness is by normalized question hash. The leftover (198-N) diamond
questions go INTO the training pool on purpose — they are the hardest,
highest-quality questions and teach the manager that hard questions warrant
deeper delegation.

Requires: accepted dataset terms on HF + `huggingface-cli login`.

Usage (from agent_routing/):
    python scripts/build_gpqa_splits.py                # eval_n=100, seed=42
    python scripts/build_gpqa_splits.py --eval_n 100 --seed 42 --out_dir outputs/data
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
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/data")
    ap.add_argument("--answer_seed", type=int, default=42)
    args = ap.parse_args()

    ext = load_gpqa(subsets="gpqa_extended", answer_seed=args.answer_seed)
    dia = load_gpqa(subsets="gpqa_diamond", answer_seed=args.answer_seed)
    if not ext or not dia:
        sys.exit("Failed to load GPQA (gated dataset — accept terms + huggingface-cli login).")

    dia_hashes = {question_hash(r.question) for r in dia}
    dia_in_ext = [r for r in ext if question_hash(r.question) in dia_hashes]
    non_dia = [r for r in ext if question_hash(r.question) not in dia_hashes]
    print(f"extended={len(ext)}  diamond={len(dia)}  matched_in_ext={len(dia_in_ext)}  non_diamond={len(non_dia)}")
    if len(dia_in_ext) < len(dia):
        print(f"WARNING: {len(dia) - len(dia_in_ext)} diamond questions not hash-matched in extended")

    rng = random.Random(args.seed)
    rng.shuffle(dia_in_ext)
    eval_rows = dia_in_ext[: args.eval_n]
    train_rows = dia_in_ext[args.eval_n :] + non_dia

    for r in eval_rows:
        r.split = "test"
    for r in train_rows:
        r.split = "train"

    # Contamination guard: eval and train must be hash-disjoint.
    eval_h = {question_hash(r.question) for r in eval_rows}
    train_h = {question_hash(r.question) for r in train_rows}
    assert not (eval_h & train_h), "eval/train overlap detected"

    os.makedirs(args.out_dir, exist_ok=True)
    eval_path = os.path.join(args.out_dir, f"gpqa_diamond_eval{len(eval_rows)}.jsonl")
    train_path = os.path.join(args.out_dir, f"gpqa_train{len(train_rows)}.jsonl")
    write_jsonl(eval_path, [r.to_dict() for r in eval_rows])
    write_jsonl(train_path, [r.to_dict() for r in train_rows])
    print(f"eval  -> {eval_path}  ({len(eval_rows)} rows, all diamond, split=test)")
    print(f"train -> {train_path}  ({len(train_rows)} rows = "
          f"{len(dia_in_ext) - len(eval_rows)} diamond + {len(non_dia)} non-diamond, split=train)")


if __name__ == "__main__":
    main()
