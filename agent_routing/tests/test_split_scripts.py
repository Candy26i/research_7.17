from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LARGE5 = (
    "corporate_lobbying",
    "definition_classification",
    "consumer_contracts_qa",
    "canada_tax_court_outcomes",
    "function_of_decision_section",
)


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


class SplitScriptTest(unittest.TestCase):
    def test_legal_large5_balanced_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in.jsonl", Path(td) / "out.jsonl"
            rows = []
            for task in LARGE5:
                for i in range(150):
                    rows.append({"example_id": len(rows), "task_subtype": task, "question": f"{task}-{i}"})
            write_jsonl(inp, rows)
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "build_legalbench_large5_splits.py"),
                "--input", str(inp), "--output", str(out),
                "--test_per_task", "100", "--dev_per_task", "40",
            ], check=True, capture_output=True, text=True)
            result = read_jsonl(out)
            split_counts = Counter(r["split"] for r in result)
            self.assertEqual(split_counts, {"test": 500, "dev": 200, "train": 50})
            per_task_test = Counter(r["task_subtype"] for r in result if r["split"] == "test")
            self.assertTrue(all(per_task_test[t] == 100 for t in LARGE5))

    def test_mmlu_custom_split_sizes(self):
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in.jsonl", Path(td) / "out.jsonl"
            rows = [{"example_id": i, "question": f"question-{i}", "metadata": {}, "split": "test"} for i in range(100)]
            write_jsonl(inp, rows)
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "build_mmlu_pro_splits.py"),
                "--input", str(inp), "--output", str(out),
                "--train_size", "50", "--dev_size", "20", "--test_size", "20",
            ], check=True, capture_output=True, text=True)
            result = read_jsonl(out)
            self.assertEqual(Counter(r["split"] for r in result), {"train": 50, "dev": 20, "test": 20})
            self.assertTrue(all(r["metadata"]["partition"] == "custom_in_domain_seeded" for r in result))

    def test_shared_synthetic_selector_uses_identical_questions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = {name: root / f"{name}.jsonl" for name in ("extractor", "reasoner", "verifier")}
            for name, path in paths.items():
                write_jsonl(path, [
                    {
                        "question_hash": f"q-{i}",
                        "task_subtype": "a" if i % 2 == 0 else "b",
                        "agent": name,
                    }
                    for i in range(12)
                ])
            out = root / "selected"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "select_shared_synthetic_rows.py"),
                "--extractor", str(paths["extractor"]),
                "--reasoner", str(paths["reasoner"]),
                "--verifier", str(paths["verifier"]),
                "--output_dir", str(out), "--n", "8",
                "--balance_by", "task_subtype", "--seed", "42",
            ], check=True, capture_output=True, text=True)
            selected = {
                name: read_jsonl(out / f"{name}_sft_final.jsonl")
                for name in paths
            }
            hash_lists = {
                name: [row["question_hash"] for row in rows]
                for name, rows in selected.items()
            }
            self.assertEqual(hash_lists["extractor"], hash_lists["reasoner"])
            self.assertEqual(hash_lists["extractor"], hash_lists["verifier"])
            self.assertEqual(Counter(r["task_subtype"] for r in selected["extractor"]), {"a": 4, "b": 4})


if __name__ == "__main__":
    unittest.main()
