"""Preflight DeepSeek prompt JSONL before paying for teacher generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter


def _read(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _question_hash(question: str) -> str:
    normalized = re.sub(r"\s+", " ", str(question).strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--min_rows", type=int, default=1)
    parser.add_argument("--require_verifier_candidates", action="store_true")
    parser.add_argument("--min_verifier_candidate_coverage", type=float, default=0.95)
    args = parser.parse_args()
    if not 0.0 <= args.min_verifier_candidate_coverage <= 1.0:
        parser.error("--min_verifier_candidate_coverage must be between 0 and 1")

    failed = False
    saw_verifier = False
    for path in args.files:
        rows = _read(path)
        agents = Counter(str(row.get("agent_kind") or "unknown") for row in rows)
        ids = [row.get("example_id") for row in rows]
        hashes = [str(row.get("question_hash") or "") for row in rows]
        duplicate_ids = len(ids) - len(set(ids))
        duplicate_hashes = len(hashes) - len(set(hashes))
        bad_hashes = 0
        bad_roles = 0
        prompt_gt_markers = 0
        verifier_rows = []
        valid_candidates = 0
        candidate_markers = 0

        for row in rows:
            if str(row.get("question_hash") or "") != _question_hash(row.get("question") or ""):
                bad_hashes += 1
            messages = row.get("prompt") or []
            roles = [str(message.get("role") or "") for message in messages]
            if roles != ["system", "user"]:
                bad_roles += 1
            prompt_text = "\n".join(str(message.get("content") or "") for message in messages)
            ground_truth = str(row.get("ground_truth") or "").strip()
            if "ground_truth" in prompt_text.lower() or (
                ground_truth and f"ANSWER_{ground_truth}" in prompt_text
            ):
                prompt_gt_markers += 1

            if row.get("agent_kind") == "verifier":
                verifier_rows.append(row)
                candidate = str(row.get("candidate_answer") or "").strip()
                choices = dict(row.get("choices") or {})
                if candidate in choices:
                    valid_candidates += 1
                    if f"CANDIDATE ANSWER TO AUDIT: {candidate}" in prompt_text:
                        candidate_markers += 1

        candidate_rate = valid_candidates / max(1, len(verifier_rows))
        marker_rate = candidate_markers / max(1, len(verifier_rows))
        print(f"\n{path}")
        print(f"  rows={len(rows)} agents={dict(agents)}")
        print(
            "  duplicate_ids="
            f"{duplicate_ids} duplicate_hashes={duplicate_hashes} "
            f"bad_hashes={bad_hashes} bad_roles={bad_roles} "
            f"prompt_gt_markers={prompt_gt_markers}"
        )
        if verifier_rows:
            saw_verifier = True
            print(
                f"  verifier_candidate_rate={candidate_rate:.3f} "
                f"candidate_marker_rate={marker_rate:.3f}"
            )

        if (
            len(rows) < args.min_rows
            or len(agents) != 1
            or duplicate_ids
            or duplicate_hashes
            or bad_hashes
            or bad_roles
            or prompt_gt_markers
        ):
            failed = True
        if args.require_verifier_candidates and verifier_rows and (
            candidate_rate < args.min_verifier_candidate_coverage
            or marker_rate < args.min_verifier_candidate_coverage
        ):
            failed = True

    if args.require_verifier_candidates and not saw_verifier:
        failed = True
        print("\nno verifier rows found")
    if failed:
        raise SystemExit("DeepSeek prompt audit failed")


if __name__ == "__main__":
    main()
