"""Smoke tests for the ADC reward: anti-collapse fixes + incentive compatibility.

Covers the tool-use-collapse fixes:
  1. Budget-truncated rollouts (dangling tool call, no final answer) are NOT
     hit with the format penalty — truncation is not a policy choice.
  2. Format violations clamp to min(r, 0) - format_penalty instead of the old
     blanket -(final_bonus + draft_bonus), so the reward floor does not deepen
     with tool count.
  3. A DRAFT_ANSWER_ line in the final turn no longer trips the answer-spam
     format check ((?<!DRAFT_) lookbehind).
  4. Bare ANSWER_ lines in intermediate turns count neither as drafts (no
     spoofing the missing-draft penalty) nor as anytime-average entries (no
     echo dilution).
  5. cost_per_tool warms up linearly over adc_cost_warmup_steps.

And the incentive-compatibility properties of the variants:
  6. "anytime" draft bonus is bounded (not farmable by extra tool calls);
     "sum" grows linearly (the exploit it exists to demonstrate).
  7. "transition" pays for a sandbagged first draft; "anytime" prefers honest.
"""
import sys

import pytest

sys.path.insert(0, ".")

from src.manager.prompt import extract_answer_sequence
from src.manager.reward import _compute_adc_reward, build_reward_funcs


KEYS = ["A", "B", "C", "D"]


def _adc_fn(**overrides):
    params = dict(
        adc_mode=True,
        adc_cost_per_tool=0.02,
        adc_draft_bonus=0.02,
        adc_missing_draft_penalty=0.1,
        adc_final_bonus=1.0,
        adc_variant="anytime",
        adc_format_penalty=0.2,
        adc_cost_warmup_steps=0,
    )
    params.update(overrides)
    return build_reward_funcs(**params)[0]


def _score(fn, completion, gt="B"):
    return fn(
        completions=[completion],
        ground_truth=[gt],
        example_id=[0],
        choice_keys=[KEYS],
    )[0]


def _tc(name="reasoner_tool"):
    return {"function": {"name": name}}


DIRECT_CORRECT = [{"role": "assistant", "content": "ANSWER_B"}]

ONE_TOOL_CORRECT = [
    {"role": "assistant", "content": "DRAFT_ANSWER_B", "tool_calls": [_tc()]},
    {"role": "tool", "content": "some tool output"},
    {"role": "assistant", "content": "ANSWER_B"},
]


def test_direct_answer_vs_one_tool_gap_is_exactly_cost():
    fn = _adc_fn()
    r0 = _score(fn, DIRECT_CORRECT)
    r1 = _score(fn, ONE_TOOL_CORRECT)
    assert r0 == pytest.approx(1.0 + 0.02)          # final + full draft avg
    assert r1 == pytest.approx(1.0 + 0.02 - 0.02)   # same, minus one tool cost
    assert r0 - r1 == pytest.approx(0.02)


def test_truncated_rollout_is_not_hammered():
    # Trajectory ends in a dangling tool call (TRL rolled the result back).
    completion = [
        {"role": "assistant", "content": "DRAFT_ANSWER_B", "tool_calls": [_tc()]},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "DRAFT_ANSWER_B", "tool_calls": [_tc("verifier_tool")]},
    ]
    r = _score(_adc_fn(), completion)
    # drafts [B, B] correct -> +0.02; two tools -> -0.04; no hammer, no final.
    assert r == pytest.approx(0.02 - 0.04)
    assert r > -0.1  # the old behavior was ~ -1.2 - costs


def test_format_violation_clamps_and_floor_does_not_deepen_with_k():
    fn = _adc_fn()
    spam_final = [{"role": "assistant", "content": "ANSWER_B\nANSWER_B"}]
    r_k0 = _score(fn, spam_final)
    assert r_k0 == pytest.approx(-0.2)  # min(1.02, 0) - 0.2

    spam_after_tools = [
        {"role": "assistant", "content": "DRAFT_ANSWER_B", "tool_calls": [_tc()]},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "ANSWER_B\nANSWER_B"},
    ]
    r_k1 = _score(fn, spam_after_tools)
    # min(1.02 - 0.02, 0) - 0.2 = -0.2: identical floor, no k-stacking beyond
    # costs already inside min(r, 0) (here r was positive pre-clamp).
    assert r_k1 == pytest.approx(-0.2)


def test_draft_line_in_final_turn_is_not_a_violation():
    completion = [{"role": "assistant", "content": "DRAFT_ANSWER_B\nANSWER_B"}]
    r = _score(_adc_fn(), completion)
    assert r > 1.0  # final bonus granted; old regex counted 2 ANSWER_ hits -> hammer


def test_bare_answer_in_intermediate_turn_is_not_a_draft():
    bare = [
        {"role": "assistant", "content": "ANSWER_B", "tool_calls": [_tc()]},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "ANSWER_B"},
    ]
    proper = ONE_TOOL_CORRECT
    fn = _adc_fn()
    r_bare = _score(fn, bare)
    r_proper = _score(fn, proper)
    # bare: no draft -> missing_draft_penalty; entries=[final] only.
    assert r_bare == pytest.approx(1.0 + 0.02 - 0.1 - 0.02)
    assert r_proper > r_bare


def test_extract_answer_sequence_ignores_intermediate_bare_answers():
    completion = [
        {"role": "assistant", "content": "ANSWER_A", "tool_calls": [_tc()]},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "ANSWER_B"},
    ]
    assert extract_answer_sequence(completion, KEYS) == ["B"]

    with_draft = [
        {"role": "assistant", "content": "DRAFT_ANSWER_A", "tool_calls": [_tc()]},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "ANSWER_B"},
    ]
    assert extract_answer_sequence(with_draft, KEYS) == ["A", "B"]


def test_cost_warmup_ramps_linearly():
    fn = _adc_fn(adc_cost_warmup_steps=10)
    # call 1: cost = 0.02 * 1/10; call 2: 0.02 * 2/10 (fallback step counter)
    r1 = _score(fn, ONE_TOOL_CORRECT)
    r2 = _score(fn, ONE_TOOL_CORRECT)
    assert r1 == pytest.approx(1.0 + 0.02 - 0.002)
    assert r2 == pytest.approx(1.0 + 0.02 - 0.004)
    assert r1 > r2


def test_anytime_bounded_sum_farmable():
    long_correct = ["B", "B", "B", "B"]   # 3 drafts + final, all correct
    short_correct = ["B", "B"]            # 1 draft + final
    common = dict(
        ground_truth="B",
        draft_bonus=0.2,
        missing_draft_penalty=0.1,
        final_bonus=1.0,
        cost_per_tool=0.0,  # isolate the draft term
        has_final=True,
    )
    r_any_long, _ = _compute_adc_reward(long_correct, n_tools=3, variant="anytime", **common)
    r_any_short, _ = _compute_adc_reward(short_correct, n_tools=1, variant="anytime", **common)
    assert r_any_long == pytest.approx(r_any_short)  # bounded: no farming

    r_sum_long, _ = _compute_adc_reward(long_correct, n_tools=3, variant="sum", **common)
    r_sum_short, _ = _compute_adc_reward(short_correct, n_tools=1, variant="sum", **common)
    assert r_sum_long > r_sum_short  # linear growth: the exploit


def test_transition_pays_sandbagging_anytime_does_not():
    common = dict(
        ground_truth="B",
        n_tools=1,
        draft_bonus=0.2,
        missing_draft_penalty=0.1,
        final_bonus=1.0,
        cost_per_tool=0.05,
        has_final=True,
    )
    sandbag = ["A", "B"]  # deliberately wrong draft, then "corrected"
    honest = ["B", "B"]
    r_tr_sandbag, _ = _compute_adc_reward(sandbag, variant="transition", **common)
    r_tr_honest, _ = _compute_adc_reward(honest, variant="transition", **common)
    assert r_tr_sandbag > r_tr_honest  # the exploit the ablation arm demonstrates

    r_any_sandbag, _ = _compute_adc_reward(sandbag, variant="anytime", **common)
    r_any_honest, _ = _compute_adc_reward(honest, variant="anytime", **common)
    assert r_any_honest > r_any_sandbag  # honest drafts optimal
