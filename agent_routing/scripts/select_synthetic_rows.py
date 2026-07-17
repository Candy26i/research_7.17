"""Select a deterministic, optionally balanced high-quality SFT subset."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--balance_by", default="", choices=["", "task_subtype", "stratum"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    seen, unique = set(), []
    for row in rows:
        key = str(row.get("question_hash") or row.get("example_id"))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    rng = random.Random(args.seed)
    if not args.balance_by:
        rng.shuffle(unique)
        chosen = unique[: args.n]
    else:
        groups = defaultdict(list)
        for row in unique:
            groups[str(row.get(args.balance_by) or "unknown")].append(row)
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        names = sorted(groups)
        while len(chosen) < args.n and any(groups[name] for name in names):
            for name in names:
                if groups[name] and len(chosen) < args.n:
                    chosen.append(groups[name].pop())

    if len(chosen) < args.n:
        raise SystemExit(f"Requested {args.n}, only {len(chosen)} unique rows available")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"selected {len(chosen)} / {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
