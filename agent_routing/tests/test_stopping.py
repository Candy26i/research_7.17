"""CPU-only unit tests for src/manager/stopping.py (no torch/transformers).

Run from the agent_routing directory:
    python -m pytest tests/test_stopping.py -q
or directly:
    python tests/test_stopping.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.manager import stopping as st  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic-record helpers
# ---------------------------------------------------------------------------

def _make_record(
    example_id: int,
    gt: str,
    stage_specs,
    k_max: int = 3,
    n_choices: int = 4,
    tools=("extractor", "reasoner", "verifier"),
):
    """stage_specs: list of (pred, votes) tuples in stage order."""
    stages = []
    for k, (pred, votes) in enumerate(stage_specs):
        stages.append({"k": k, "pred": pred, "votes": list(votes)})
    steps = []
    for i in range(len(stages) - 1):
        kind = tools[i % len(tools)]
        steps.append({
            "i": i,
            "tool_name": f"{kind}_tool",
            "kind": kind,
            "chosen_by": "model",
            "assistant_content": f"DRAFT_ANSWER_{stages[i]['pred']}" if stages[i]["pred"] else "",
            "args": {"example_id": example_id},
            "call_id": f"cf_{example_id}_{i}",
            "tool_output": json.dumps({"note": f"tool output {i}"}),
        })
    return {
        "example_id": example_id,
        "question_hash": f"hash_{example_id}",
        "ground_truth": gt,
        "choice_keys": ["A", "B", "C", "D"][:n_choices],
        "n_choices": n_choices,
        "question_len": 120,
        "context_len": 40,
        "k_max": k_max,
        "base_messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"question {example_id}"},
        ],
        "stages": stages,
        "steps": steps,
    }


def _confident_votes(key, n=5):
    return [key] * n


def _split_votes(a, b, n_a=3, n_b=2):
    return [a] * n_a + [b] * n_b


def _synthetic_dataset(n=120):
    """Correctness is (noisily) determined by vote agreement so the probe has
    real signal to find: unanimous stages are correct, split stages wrong."""
    records = []
    for i in range(n):
        if i % 2 == 0:
            # Easy question: correct and unanimous from stage 0.
            specs = [("A", _confident_votes("A"))] * 4
            records.append(_make_record(i, "A", specs))
        else:
            # Hard question: wrong/split until stage 2, then correct+unanimous.
            specs = [
                ("B", _split_votes("B", "C")),
                ("C", _split_votes("C", "B")),
                ("A", _confident_votes("A")),
                ("A", _confident_votes("A")),
            ]
            records.append(_make_record(i, "A", specs))
    return records


# ---------------------------------------------------------------------------
# vote_stats / stage_features
# ---------------------------------------------------------------------------

def test_vote_stats_empty_and_invalid():
    vs = st.vote_stats([], 4)
    assert vs["majority"] is None
    assert vs["agreement_top"] == 0.0
    assert vs["valid_vote_frac"] == 0.0
    assert vs["vote_entropy_norm"] == 1.0

    vs = st.vote_stats([None, None], 4)
    assert vs["majority"] is None and vs["valid_vote_frac"] == 0.0


def test_vote_stats_majority_tiebreak_and_entropy():
    # Tie between A and B -> deterministic pick "A" (count desc, key asc).
    vs = st.vote_stats(["B", "A", "B", "A"], 4)
    assert vs["majority"] == "A"
    assert abs(vs["agreement_top"] - 0.5) < 1e-9
    # Two equally likely outcomes: entropy = log2 / log4 = 0.5.
    assert abs(vs["vote_entropy_norm"] - (math.log(2) / math.log(4))) < 1e-9

    vs = st.vote_stats(["A", "A", "A", None], 4)
    assert vs["majority"] == "A"
    assert abs(vs["agreement_top"] - 1.0) < 1e-9
    assert abs(vs["valid_vote_frac"] - 0.75) < 1e-9
    assert vs["vote_entropy_norm"] == 0.0


def test_stage_features_order_and_values():
    feats = st.stage_features(
        k=1, k_max=3, greedy_pred="A", votes=["A", "A", "B"],
        prev_pred="A", n_choices=4, question_len=100, context_len=0,
    )
    assert len(feats) == len(st.FEATURE_NAMES)
    named = dict(zip(st.FEATURE_NAMES, feats))
    assert abs(named["k_norm"] - 1 / 3) < 1e-9
    assert named["is_k0"] == 0.0
    assert abs(named["agreement_top"] - 2 / 3) < 1e-9
    assert named["greedy_matches_majority"] == 1.0
    assert named["stable_from_prev"] == 1.0
    assert abs(named["chance_rate"] - 0.25) < 1e-9
    assert named["log_ctxlen"] == 0.0

    # k=0: never "stable", is_k0 set.
    feats0 = st.stage_features(
        k=0, k_max=3, greedy_pred="A", votes=["A"], prev_pred=None,
        n_choices=4, question_len=100, context_len=0,
    )
    named0 = dict(zip(st.FEATURE_NAMES, feats0))
    assert named0["is_k0"] == 1.0 and named0["stable_from_prev"] == 0.0


def test_record_stage_feature_rows_prev_pred_chain():
    rec = _make_record(1, "A", [
        ("B", ["B"]), ("B", ["B"]), ("A", ["A"]), ("A", ["A"]),
    ])
    rows = st.record_stage_feature_rows(rec)
    assert len(rows) == 4
    stable_idx = st.FEATURE_NAMES.index("stable_from_prev")
    # Stage 1 pred B == stage 0 pred B -> stable; stage 2 pred A != B -> not.
    assert rows[0]["features"][stable_idx] == 0.0
    assert rows[1]["features"][stable_idx] == 1.0
    assert rows[2]["features"][stable_idx] == 0.0
    assert rows[3]["features"][stable_idx] == 1.0
    assert [r["label"] for r in rows] == [0.0, 0.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Probe fitting, prediction consistency, save/load
# ---------------------------------------------------------------------------

def test_fit_probe_learns_agreement_signal():
    records = _synthetic_dataset()
    probe, metrics = st.fit_stop_probe(records, l2=1e-3)
    assert metrics["auc"] is not None and metrics["auc"] > 0.9

    confident = st.stage_features(0, 3, "A", _confident_votes("A"), None, 4, 120, 40)
    split = st.stage_features(0, 3, "B", _split_votes("B", "C"), None, 4, 120, 40)
    p_conf = probe.predict_proba(confident)
    p_split = probe.predict_proba(split)
    assert p_conf > 0.8
    assert p_split < 0.5
    assert p_conf > p_split


def test_probe_save_load_roundtrip_and_schema_guard():
    records = _synthetic_dataset(40)
    probe, _ = st.fit_stop_probe(records)
    probe.threshold = 0.77
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "probe.json")
        probe.save(path)
        loaded = st.StopProbe.load(path)
        feats = st.stage_features(1, 3, "A", ["A", "A"], "A", 4, 100, 10)
        assert abs(loaded.predict_proba(feats) - probe.predict_proba(feats)) < 1e-12
        assert loaded.threshold == 0.77

        # Feature-schema mismatch must refuse to load.
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["feature_names"] = list(reversed(d["feature_names"]))
        try:
            st.StopProbe.from_dict(d)
            assert False, "expected ValueError on feature schema mismatch"
        except ValueError:
            pass


def test_predict_proba_wrong_arity_raises():
    records = _synthetic_dataset(40)
    probe, _ = st.fit_stop_probe(records)
    try:
        probe.predict_proba([0.1, 0.2])
        assert False, "expected ValueError on wrong feature count"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Threshold sweep / policy simulation / baselines
# ---------------------------------------------------------------------------

def test_sweep_extremes_and_choice():
    records = _synthetic_dataset()
    probe, _ = st.fit_stop_probe(records)

    sweep = st.threshold_sweep(records, probe)
    by_thr = {pt["threshold"]: pt for pt in sweep}

    # theta = 0: stop immediately everywhere -> avg 0 tools, acc = fixed-k(0).
    fixed = st.fixed_k_stats(records, k_max=3)
    assert by_thr[0.0]["avg_tool_calls"] == 0.0
    assert abs(by_thr[0.0]["accuracy"] - fixed["0"]["accuracy"]) < 1e-9

    # sentinel 1.01: never stop early -> always k_max, acc = fixed-k(3).
    assert by_thr[1.01]["avg_tool_calls"] == 3.0
    assert abs(by_thr[1.01]["accuracy"] - fixed["3"]["accuracy"]) < 1e-9

    acc_full = fixed["3"]["accuracy"]
    choice = st.choose_threshold(sweep, acc_full, epsilon=0.005)
    assert choice["constraint_satisfied"]
    assert choice["accuracy"] >= acc_full - 0.005
    # On this dataset easy halves are solvable at k=0 and hard halves at k=2,
    # so a good probe must save tools vs always-3.
    assert choice["avg_tool_calls"] < 3.0

    # Impossible constraint falls back and flags it.
    fallback = st.choose_threshold(sweep, acc_full=2.0, epsilon=0.0)
    assert not fallback["constraint_satisfied"]


def test_fixed_k_oracle_heuristics():
    records = [
        _make_record(0, "A", [
            ("A", _confident_votes("A")), ("A", _confident_votes("A")),
            ("A", _confident_votes("A")), ("A", _confident_votes("A")),
        ]),
        _make_record(1, "A", [
            ("B", _split_votes("B", "C")), ("B", _split_votes("B", "A")),
            ("A", _confident_votes("A")), ("B", _split_votes("B", "A")),
        ]),
        _make_record(2, "A", [
            ("C", _split_votes("C", "D")), ("C", _split_votes("C", "D")),
            ("C", _split_votes("C", "D")), ("C", _split_votes("C", "D")),
        ]),
    ]
    # Reported stats are rounded to 4 decimals -> compare with 1e-3 tolerance.
    fixed = st.fixed_k_stats(records, k_max=3)
    assert abs(fixed["0"]["accuracy"] - 1 / 3) < 1e-3
    assert abs(fixed["2"]["accuracy"] - 2 / 3) < 1e-3
    assert abs(fixed["3"]["accuracy"] - 1 / 3) < 1e-3  # record 1 regresses at k=3

    oracle = st.oracle_stats(records)
    assert abs(oracle["accuracy"] - 2 / 3) < 1e-3
    # Record 0 stops at 0, record 1 at 2, record 2 never correct (0 tools).
    assert abs(oracle["avg_tool_calls"] - (0 + 2 + 0) / 3) < 1e-3

    heur = st.heuristic_stats(records)
    assert set(heur.keys()) == {"stop_on_stable", "stop_on_unanimous"}
    # stop_on_stable: rec0 stops at 1 (A==A); rec1 stops at 1 (B==B, wrong);
    # rec2 stops at 1 (C==C, wrong).
    assert abs(heur["stop_on_stable"]["avg_tool_calls"] - 1.0) < 1e-3
    assert abs(heur["stop_on_stable"]["accuracy"] - 1 / 3) < 1e-3


def test_marginal_tool_table():
    records = [
        _make_record(0, "A", [
            ("B", ["B"]), ("A", ["A"]), ("A", ["A"]), ("A", ["A"]),
        ]),  # extractor: gain; reasoner: none; verifier: none
        _make_record(1, "A", [
            ("A", ["A"]), ("B", ["B"]), ("B", ["B"]), ("B", ["B"]),
        ]),  # extractor: loss
    ]
    table = st.marginal_tool_table(records)
    assert table["extractor"]["n_steps"] == 2
    assert abs(table["extractor"]["gain_rate"] - 0.5) < 1e-9
    assert abs(table["extractor"]["loss_rate"] - 0.5) < 1e-9
    assert abs(table["extractor"]["net_gain"] - 0.0) < 1e-9
    assert table["reasoner"]["net_gain"] == 0.0


def test_stable_holdout_split_deterministic_partition():
    records = _synthetic_dataset(100)
    a_train, a_hold = st.stable_holdout_split(records, 0.25)
    b_train, b_hold = st.stable_holdout_split(records, 0.25)
    assert [r["example_id"] for r in a_train] == [r["example_id"] for r in b_train]
    assert [r["example_id"] for r in a_hold] == [r["example_id"] for r in b_hold]
    assert len(a_train) + len(a_hold) == 100
    assert 5 <= len(a_hold) <= 45  # crc32 buckets are roughly uniform
    train_ids = {r["example_id"] for r in a_train}
    hold_ids = {r["example_id"] for r in a_hold}
    assert not (train_ids & hold_ids)


# ---------------------------------------------------------------------------
# Trajectory hygiene + distillation export
# ---------------------------------------------------------------------------

def test_answer_to_draft():
    text = "Some reasoning.\nANSWER_B"
    assert st.answer_to_draft(text) == "Some reasoning.\nDRAFT_ANSWER_B"
    # Indented ANSWER lines are rewritten; DRAFT lines untouched.
    text2 = "  ANSWER_C\nDRAFT_ANSWER_A\nmid ANSWER_B stays"
    out2 = st.answer_to_draft(text2)
    assert out2.splitlines()[0] == "  DRAFT_ANSWER_C"
    assert out2.splitlines()[1] == "DRAFT_ANSWER_A"
    assert out2.splitlines()[2] == "mid ANSWER_B stays"
    assert st.answer_to_draft("") == ""


def test_pick_stop_index_and_distill_rows():
    records = _synthetic_dataset()
    probe, _ = st.fit_stop_probe(records)
    fixed = st.fixed_k_stats(records, k_max=3)
    sweep = st.threshold_sweep(records, probe)
    thr = st.choose_threshold(sweep, fixed["3"]["accuracy"], 0.005)["threshold"]

    easy = records[0]   # correct + unanimous from stage 0
    hard = records[1]   # correct + unanimous from stage 2
    assert st.pick_stop_index(easy, probe, thr) == 0
    assert st.pick_stop_index(hard, probe, thr) == 2

    # Easy: k*=0 -> a single final-answer row, no tool turns.
    rows = st.build_distill_rows(easy, 0)
    assert len(rows) == 1
    assert rows[0]["prompt"] == easy["base_messages"]
    assert rows[0]["response"][0]["content"] == "ANSWER_A"

    # Hard: k*=2 -> two tool-call turns + final turn with draft+answer.
    rows = st.build_distill_rows(hard, 2)
    assert len(rows) == 3
    call0 = rows[0]["response"][0]
    assert call0["role"] == "assistant"
    fn = call0["tool_calls"][0]["function"]
    assert fn["name"] == "extractor_tool"
    assert json.loads(fn["arguments"]) == {"example_id": hard["example_id"]}
    # Second turn's prompt contains the first tool exchange.
    assert rows[1]["prompt"][-1]["role"] == "tool"
    assert rows[1]["prompt"][-2]["role"] == "assistant"
    # Final turn: draft + answer for the (correct) stage-2 prediction.
    assert rows[2]["response"][0]["content"] == "DRAFT_ANSWER_A\nANSWER_A"
    assert len(rows[2]["prompt"]) == 2 + 2 * 2  # base + 2 (call, tool) pairs


def test_distill_rejects_wrong_or_missing_answers():
    wrong = _make_record(7, "A", [
        ("B", ["B"]), ("B", ["B"]), ("B", ["B"]), ("B", ["B"]),
    ])
    assert st.build_distill_rows(wrong, 2) == []
    assert len(st.build_distill_rows(wrong, 2, require_correct=False)) == 3

    missing = _make_record(8, "A", [
        (None, []), ("A", ["A"]), ("A", ["A"]), ("A", ["A"]),
    ])
    assert st.build_distill_rows(missing, 0) == []
    # k*=1 works: the stage-0 draft content falls back to empty -> the
    # intermediate assistant turn simply has no content key.
    rows = st.build_distill_rows(missing, 1)
    assert len(rows) == 2
    assert "content" not in rows[0]["response"][0]

    # k_star beyond recorded stages clamps to the last stage.
    short = _make_record(9, "A", [("A", ["A"]), ("A", ["A"])])
    rows = st.build_distill_rows(short, 5)
    assert len(rows) == 2  # clamped to k*=1: one tool turn + final


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
