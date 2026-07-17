"""Select one shared question set across extractor/reasoner/verifier SFT files."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {str(r.get("question_hash") or ""): r for r in rows if r.get("question_hash")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", required=True)
    ap.add_argument("--reasoner", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--balance_by", default="", choices=["", "task_subtype", "stratum"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    maps = {
        "extractor": read(args.extractor),
        "reasoner": read(args.reasoner),
        "verifier": read(args.verifier),
    }
    shared = set.intersection(*(set(m) for m in maps.values()))
    rng = random.Random(args.seed)
    if not args.balance_by:
        candidates = sorted(shared)
        rng.shuffle(candidates)
        chosen = candidates[: args.n]
    else:
        groups = defaultdict(list)
        reference = maps["extractor"]
        for h in shared:
            groups[str(reference[h].get(args.balance_by) or "unknown")].append(h)
        for group in groups.values():
            rng.shuffle(group)
        chosen, names = [], sorted(groups)
        while len(chosen) < args.n and any(groups[name] for name in names):
            for name in names:
                if groups[name] and len(chosen) < args.n:
                    chosen.append(groups[name].pop())

    if len(chosen) < args.n:
        raise SystemExit(
            f"Need {args.n} shared accepted questions; only {len(shared)} available. "
            "Generate a larger raw pool."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    for agent, mapping in maps.items():
        path = os.path.join(args.output_dir, f"{agent}_sft_final.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for h in chosen:
                f.write(json.dumps(mapping[h], ensure_ascii=False) + "\n")
        print(f"{agent}: {len(chosen)} shared rows -> {path}")


if __name__ == "__main__":
    main()
