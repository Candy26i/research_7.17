from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from src.verifier_candidates import (
    load_validated_prediction_map,
    require_verifier_candidate_coverage,
    validate_candidate_bound_prompt_rows,
)


@dataclass
class Row:
    example_id: int
    question: str
    choices: dict


def _hash(question: str) -> str:
    import hashlib
    import re

    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def _rows():
    return [
        Row(index, f"Question {index}?", {"A": "alpha", "B": "beta"})
        for index in range(4)
    ]


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class VerifierCandidateSafetyTest(unittest.TestCase):
    def test_pipeline_wires_strict_checks_by_default(self):
        root = Path(__file__).resolve().parents[1]
        stages = (root / "src" / "pipeline" / "stages.py").read_text(encoding="utf-8")
        cli = (root / "src" / "pipeline" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("require_verifier_candidate_coverage(", stages)
        self.assertIn("validate_candidate_bound_prompt_rows(prompt_rows)", stages)
        self.assertIn("--synth_allow_empty_verifier_candidates", cli)

    def test_load_and_coverage_accept_matching_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = _rows()
            _write_jsonl(path, [
                {"example_id": row.example_id, "question_hash": _hash(row.question), "pred": "B"}
                for row in rows
            ])
            candidate_map = load_validated_prediction_map(str(path), rows)
            stats = require_verifier_candidate_coverage(rows, candidate_map)
            self.assertEqual(candidate_map, {0: "B", 1: "B", 2: "B", 3: "B"})
            self.assertEqual(stats["candidate_coverage"], 1.0)

    def test_low_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 95%"):
            require_verifier_candidate_coverage(_rows(), {0: "B"})

    def test_missing_or_mismatched_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = _rows()
            _write_jsonl(path, [{"example_id": 0, "pred": "B"}])
            with self.assertRaisesRegex(ValueError, "missing question_hash"):
                load_validated_prediction_map(str(path), rows)
            _write_jsonl(path, [{"example_id": 0, "question_hash": "wrong", "pred": "B"}])
            with self.assertRaisesRegex(ValueError, "question_hash mismatch"):
                load_validated_prediction_map(str(path), rows)

    def test_prompt_rows_must_be_candidate_bound(self):
        row = _rows()[0]
        base = {
            "example_id": row.example_id,
            "agent_kind": "verifier",
            "question": row.question,
            "question_hash": _hash(row.question),
            "choices": row.choices,
            "candidate_answer": "",
            "prompt": [{"role": "user", "content": "question"}],
        }
        with self.assertRaisesRegex(ValueError, "candidate-bound prompts"):
            validate_candidate_bound_prompt_rows([base])

        candidate_bound = dict(base)
        candidate_bound["candidate_answer"] = "B"
        candidate_bound["prompt"] = [
            {"role": "system", "content": "verify"},
            {"role": "user", "content": "CANDIDATE ANSWER TO AUDIT: B"},
        ]
        validate_candidate_bound_prompt_rows([candidate_bound])

    def test_prompt_audit_accepts_bound_and_rejects_empty_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(__file__).resolve().parents[1]
            path = Path(directory) / "verifier_prompts.jsonl"
            row = _rows()[0]
            prompt_row = {
                "example_id": row.example_id,
                "agent_kind": "verifier",
                "question": row.question,
                "question_hash": _hash(row.question),
                "choices": row.choices,
                "ground_truth": "A",
                "candidate_answer": "B",
                "prompt": [
                    {"role": "system", "content": "verify"},
                    {"role": "user", "content": "CANDIDATE ANSWER TO AUDIT: B"},
                ],
            }
            _write_jsonl(path, [prompt_row])
            command = [
                sys.executable,
                str(root / "scripts" / "audit_deepseek_prompts.py"),
                str(path),
                "--require_verifier_candidates",
                "--min_verifier_candidate_coverage",
                "1.0",
            ]
            accepted = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)

            prompt_row["candidate_answer"] = ""
            _write_jsonl(path, [prompt_row])
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
