"""Build one frozen custom in-domain MMLU-Pro partition.

MMLU-Pro's public corpus is the official test pool. This script deliberately
creates a non-standard, question-disjoint partition for in-domain policy
training. Results must not be presented as standard leaderboard performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random


def _read(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hash(row) -> str:
    text = " ".join(str(row.get("question") or "").lower().split())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Normalized 12,032-row MMLU-Pro JSONL")
    ap.add_argument("--output", default="outputs/data/mmlu_pro_custom_split.jsonl")
    ap.add_argument("--train_size", type=int, default=4000)
    ap.add_argument("--dev_size", type=int, default=300)
    ap.add_argument("--test_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = _read(args.input)
    need = args.train_size + args.dev_size + args.test_size
    if len(rows) < need:
        raise SystemExit(f"Need {need} rows, found {len(rows)}")
    random.Random(args.seed).shuffle(rows)
    test = rows[: args.test_size]
    dev = rows[args.test_size:args.test_size + args.dev_size]
    train = rows[args.test_size + args.dev_size:need]

    for split, part in (("train", train), ("dev", dev), ("test", test)):
        for row in part:
            metadata = dict(row.get("metadata") or {})
            metadata["original_split"] = row.get("split", "test")
            metadata["partition"] = "custom_in_domain_seeded"
            row["metadata"] = metadata
            row["split"] = split

    hs = [{_hash(r) for r in part} for part in (train, dev, test)]
    assert not (hs[0] & hs[1] or hs[0] & hs[2] or hs[1] & hs[2])
    _write(args.output, train + dev + test)
    print(
        f"MMLU-Pro custom split train/dev/test="
        f"{len(train)}/{len(dev)}/{len(test)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
