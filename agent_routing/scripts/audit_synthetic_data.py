"""Audit sub-agent synthetic JSONL before any SFT run."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min_rows", type=int, default=1)
    ap.add_argument("--max_duplicate_rate", type=float, default=0.02)
    ap.add_argument("--require_verifier_candidates", action="store_true")
    args = ap.parse_args()

    hashes_by_agent = defaultdict(set)
    failed = False
    for path in args.files:
        rows = _read(path)
        agents = Counter(str(r.get("agent_kind") or "unknown") for r in rows)
        tasks = Counter(str(r.get("task_subtype") or r.get("stratum") or "unknown") for r in rows)
        hashes = [str(r.get("question_hash") or "") for r in rows]
        nonempty = [h for h in hashes if h]
        duplicate_rate = 1.0 - len(set(nonempty)) / max(1, len(nonempty))
        json_ok = 0
        for row in rows:
            response = row.get("response")
            if isinstance(response, dict):
                json_ok += 1
            elif isinstance(response, str):
                try:
                    json_ok += int(isinstance(json.loads(response), dict))
                except Exception:
                    pass
        verifier_rows = [r for r in rows if r.get("agent_kind") == "verifier"]
        candidate_rate = sum(bool(r.get("candidate_answer")) for r in verifier_rows) / max(1, len(verifier_rows))
        candidate_correct = sum(bool(r.get("candidate_correct")) for r in verifier_rows) / max(1, len(verifier_rows))
        print(f"\n{path}")
        print(f"  rows={len(rows)} agents={dict(agents)} json_ok={json_ok/max(1,len(rows)):.3f}")
        print(f"  duplicate_question_rate={duplicate_rate:.3f} tasks={dict(tasks)}")
        if verifier_rows:
            print(f"  verifier_candidate_rate={candidate_rate:.3f} candidate_correct_rate={candidate_correct:.3f}")
        if len(rows) < args.min_rows or duplicate_rate > args.max_duplicate_rate:
            failed = True
        if args.require_verifier_candidates and verifier_rows and candidate_rate < 0.95:
            failed = True
        for agent in agents:
            hashes_by_agent[agent].update(nonempty)

    names = sorted(hashes_by_agent)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            inter = hashes_by_agent[left] & hashes_by_agent[right]
            union = hashes_by_agent[left] | hashes_by_agent[right]
            print(f"overlap {left}/{right}: n={len(inter)} jaccard={len(inter)/max(1,len(union)):.3f}")
    if failed:
        raise SystemExit("Synthetic-data audit failed")


if __name__ == "__main__":
    main()
