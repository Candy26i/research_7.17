"""Create a task-balanced LegalBench large5 train/dev/test partition.

The final test pool contains exactly ``test_per_task`` rows from each config;
dev is balanced the same way and every remaining row becomes training data.
The combined output can be passed directly as --legalbench_normalized_cache.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict


LARGE5 = (
    "corporate_lobbying",
    "definition_classification",
    "consumer_contracts_qa",
    "canada_tax_court_outcomes",
    "function_of_decision_section",
)


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _task(row) -> str:
    return str(
        row.get("task_subtype")
        or (row.get("metadata") or {}).get("legalbench_config")
        or ""
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Normalized large5 JSONL")
    ap.add_argument("--output", default="outputs/data/legalbench_large5_split.jsonl")
    ap.add_argument("--test_per_task", type=int, default=100,
                    help="100 x 5 creates a 500-row held-out pool")
    ap.add_argument("--dev_per_task", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    groups = defaultdict(list)
    for row in _read_jsonl(args.input):
        groups[_task(row)].append(row)

    missing = [task for task in LARGE5 if task not in groups]
    if missing:
        raise SystemExit(f"Missing large5 tasks: {missing}; found={sorted(groups)}")

    train, dev, test = [], [], []
    for offset, task in enumerate(LARGE5):
        rows = list(groups[task])
        random.Random(args.seed + offset).shuffle(rows)
        need = args.test_per_task + args.dev_per_task
        if len(rows) < need:
            raise SystemExit(
                f"{task} has {len(rows)} rows, fewer than required {need}"
            )
        task_test = rows[: args.test_per_task]
        task_dev = rows[args.test_per_task:need]
        task_train = rows[need:]
        for row in task_test:
            row["split"] = "test"
        for row in task_dev:
            row["split"] = "dev"
        for row in task_train:
            row["split"] = "train"
        test.extend(task_test)
        dev.extend(task_dev)
        train.extend(task_train)
        print(f"{task:32s} train/dev/test={len(task_train)}/{len(task_dev)}/{len(task_test)}")

    combined = train + dev + test
    _write_jsonl(args.output, combined)
    print(f"TOTAL train/dev/test={len(train)}/{len(dev)}/{len(test)} -> {args.output}")


if __name__ == "__main__":
    main()
