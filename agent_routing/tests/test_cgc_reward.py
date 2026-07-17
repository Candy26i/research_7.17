"""Tests for CGC (Design A — Counterfactual Group Composition) reward.

Covers:
  1. Off-arm rollouts (tool calls answered with the tools_unavailable
     sentinel) are pure binary: no tool cost, no missing-draft penalty.
  2. On-arm cost is charged per EXECUTED tool (k_eff), not per attempt.
  3. Missing-draft penalty is per tool-calling TURN (per-turn pairing —
     the ADC_RESIDUAL_HOLES §2 fix): post-hoc drafts cannot satisfy it.
  4. Group flattening ("novar"): groups with no correctness variance get
     all rewards set to the group mean (advantage 0); mixed groups keep
     their cost differences as a parsimony tiebreaker.
  5. Truncated rollouts (dangling tool call) are exempt from the format
     penalty, same as ADC.
  6. Telemetry invariants via count_unpaired_tool_turns.
"""
import sys

import pytest

sys.path.insert(0, ".")

from src.manager.prompt import count_unpaired_tool_turns
from src.manager.reward import TOOLS_UNAVAILABLE_MSG, build_reward_funcs


KEYS = ["A", "B", "C", "D"]


def _cgc_fn(**overrides):
    params = dict(
        cgc_mode=True,
        cgc_cost_per_tool=0.01,
        cgc_missing_draft_penalty=0.05,
        cgc_cost_warmup_steps=0,
        cgc_flatten="none",
        adc_format_penalty=0.2,
    )
    params.update(overrides)
    return build_reward_funcs(**params)[0]


def _score_batch(fn, completions, gts, eids):
    return fn(
        completions=completions,
        ground_truth=gts,
        example_id=eids,
        choice_keys=[KEYS] * len(completions),
    )


def _tc(name="reasoner_tool"):
    return {"function": {"name": name}}


def _on_arm(correct=True, k=1, with_draft=True):
    msgs = []
    for _ in range(k):
        turn = {"role": "assistant", "tool_calls": [_tc()]}
        turn["content"] = "DRAFT_ANSWER_B" if with_draft else ""
        msgs.append(turn)
        msgs.append({"role": "tool", "content": '{"analysis": "real tool output"}'})
    msgs.append({"role": "assistant", "content": "ANSWER_B" if correct else "ANSWER_C"})
    return msgs


def _off_arm_blocked(correct=True):
    return [
        {"role": "assistant", "content": "DRAFT_ANSWER_B", "tool_calls": [_tc()]},
        {"role": "tool", "content": TOOLS_UNAVAILABLE_MSG},
        {"role": "assistant", "content": "ANSWER_B" if correct else "ANSWER_C"},
    ]


def _direct(correct=True):
    return [{"role": "assistant", "content": "ANSWER_B" if correct else "ANSWER_C"}]


def test_off_arm_is_pure_binary():
    fn = _cgc_fn()
    r = _score_batch(fn, [_off_arm_blocked(correct=True)], ["B"], [0])[0]
    assert r == pytest.approx(1.0)  # no cost for the blocked attempt
    r_wrong = _score_batch(fn, [_off_arm_blocked(correct=False)], ["B"], [0])[0]
    assert r_wrong == pytest.approx(0.0)


def test_off_arm_without_draft_not_penalized():
    completion = [
        {"role": "assistant", "content": "", "tool_calls": [_tc()]},
        {"role": "tool", "content": TOOLS_UNAVAILABLE_MSG},
        {"role": "assistant", "content": "ANSWER_B"},
    ]
    r = _score_batch(_cgc_fn(), [completion], ["B"], [0])[0]
    assert r == pytest.approx(1.0)  # the harness said no; not a policy failure


def test_on_arm_cost_per_executed_tool():
    fn = _cgc_fn()
    r1 = _score_batch(fn, [_on_arm(correct=True, k=1)], ["B"], [0])[0]
    r2 = _score_batch(fn, [_on_arm(correct=True, k=2)], ["B"], [0])[0]
    assert r1 == pytest.approx(1.0 - 0.01)
    assert r2 == pytest.approx(1.0 - 0.02)


def test_missing_draft_is_per_turn_and_posthoc_does_not_satisfy():
    fn = _cgc_fn()
    # 2 tool turns, no drafts at all -> 2 unpaired turns.
    r_none = _score_batch(fn, [_on_arm(correct=True, k=2, with_draft=False)], ["B"], [0])[0]
    assert r_none == pytest.approx(1.0 - 0.02 - 2 * 0.05)
    # Post-hoc draft in the final turn does NOT pair the tool turns.
    posthoc = _on_arm(correct=True, k=2, with_draft=False)
    posthoc[-1]["content"] = "DRAFT_ANSWER_B\nANSWER_B"
    r_posthoc = _score_batch(fn, [posthoc], ["B"], [0])[0]
    assert r_posthoc == pytest.approx(1.0 - 0.02 - 2 * 0.05)
    # Proper per-turn drafts -> no penalty.
    r_paired = _score_batch(fn, [_on_arm(correct=True, k=2, with_draft=True)], ["B"], [0])[0]
    assert r_paired == pytest.approx(1.0 - 0.02)


def test_flatten_novar_zeroes_uniform_groups():
    fn = _cgc_fn(cgc_flatten="novar")
    # One group (same eid), all correct, mixed tool usage -> flattened equal.
    comps = [_direct(True), _direct(True), _on_arm(True, k=1), _on_arm(True, k=2)]
    rs = _score_batch(fn, comps, ["B"] * 4, [7, 7, 7, 7])
    assert len(set(round(r, 6) for r in rs)) == 1  # all equal -> advantage 0
    assert rs[0] == pytest.approx((1.0 + 1.0 + 0.99 + 0.98) / 4)


def test_flatten_novar_keeps_mixed_groups():
    fn = _cgc_fn(cgc_flatten="novar")
    comps = [_direct(True), _direct(False), _on_arm(True, k=1), _on_arm(False, k=1)]
    rs = _score_batch(fn, comps, ["B"] * 4, [7, 7, 7, 7])
    assert rs[0] == pytest.approx(1.0)
    assert rs[1] == pytest.approx(0.0)
    assert rs[2] == pytest.approx(0.99)   # cost survives as parsimony tiebreaker
    assert rs[3] == pytest.approx(-0.01)


def test_flatten_groups_by_example_id_not_position():
    fn = _cgc_fn(cgc_flatten="novar")
    # Interleaved eids: group 1 uniform (flattened), group 2 mixed (kept).
    comps = [_direct(True), _direct(True), _on_arm(True, k=1), _direct(False)]
    rs = _score_batch(fn, comps, ["B"] * 4, [1, 2, 1, 2])
    assert rs[0] == pytest.approx(rs[2])                  # group 1 flattened
    assert rs[1] == pytest.approx(1.0)                    # group 2 untouched
    assert rs[3] == pytest.approx(0.0)


def test_truncated_rollout_exempt_from_format_penalty():
    completion = [
        {"role": "assistant", "content": "DRAFT_ANSWER_B", "tool_calls": [_tc()]},
        {"role": "tool", "content": '{"analysis": "output"}'},
        {"role": "assistant", "content": "DRAFT_ANSWER_B", "tool_calls": [_tc("verifier_tool")]},
    ]
    r = _score_batch(_cgc_fn(), [completion], ["B"], [0])[0]
    assert r > -0.1  # ~ -cost*k_eff, never the -0.2 format hammer


def test_format_violation_still_clamped():
    spam = [{"role": "assistant", "content": "ANSWER_B\nANSWER_B"}]
    r = _score_batch(_cgc_fn(), [spam], ["B"], [0])[0]
    assert r == pytest.approx(-0.2)


def test_cost_warmup_applies():
    fn = _cgc_fn(cgc_cost_warmup_steps=10)
    r1 = _score_batch(fn, [_on_arm(True, k=1)], ["B"], [0])[0]
    r2 = _score_batch(fn, [_on_arm(True, k=1)], ["B"], [0])[0]
    assert r1 == pytest.approx(1.0 - 0.001)
    assert r2 == pytest.approx(1.0 - 0.002)


def test_count_unpaired_tool_turns():
    assert count_unpaired_tool_turns(_on_arm(True, k=2, with_draft=True), KEYS) == (2, 0)
    assert count_unpaired_tool_turns(_on_arm(True, k=2, with_draft=False), KEYS) == (2, 2)
    assert count_unpaired_tool_turns(_direct(True), KEYS) == (0, 0)
