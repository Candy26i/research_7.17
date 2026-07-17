"""Summarize belief transitions in generated cold-start SFT trajectories."""
from __future__ import annotations

import argparse
import json
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    args = ap.parse_args()
    with open(args.jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    finals = [r for r in rows if isinstance(r.get("draft_sequence"), list)]
    transitions = Counter()
    depths = Counter()
    for row in finals:
        gt = str(row.get("ground_truth") or "")
        seq = [str(x) for x in row.get("draft_sequence") or []]
        # draft_sequence contains the initial belief plus one belief after
        # each tool result, so its transition count is the executed depth.
        depths[max(0, len(seq) - 1)] += 1
        for left, right in zip(seq, seq[1:]):
            transitions[("C" if left == gt else "W") + "->" + ("C" if right == gt else "W")] += 1
    print(f"examples={len(finals)} depths={dict(sorted(depths.items()))}")
    print(f"transitions={dict(transitions)}")
    if not finals:
        raise SystemExit("No terminal cold-start rows with draft_sequence found")


if __name__ == "__main__":
    main()
