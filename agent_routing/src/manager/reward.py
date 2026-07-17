"""Reward functions for manager GRPO.

Three reward modes:

1. Binary (default, ccr_mode=False, adc_mode=False):
     R = 1.0 if correct else 0.0
     + optional routing_efficiency_bonus * saved_tool_calls (when correct)
     + optional tool_use_bonus (when correct and >=1 tool called)

2. CCR — Calibrated Confidence Routing (ccr_mode=True):  [LEGACY, broken for k>=2]
     Frames routing as an implicit confidence claim and applies the log scoring rule.
     WARNING: when p_low < 0.5, rewards for k>=2 are INVERTED (wrong > correct).
     Only use with p_low > 0.5 (e.g. 0.6).

3. ADC — Adaptive Deliberation Control (adc_mode=True):  [RECOMMENDED]
     Anytime-accuracy reward. Manager outputs DRAFT_ANSWER_<TOKEN> in every
     turn that calls a tool; reward is:
       +draft_bonus * (fraction of answer statements that are correct)
       -missing_draft_penalty  per tool call not accompanied by a draft
       +final_bonus            if the final submitted answer is correct
       -cost_per_tool          per tool called (encourages stopping early)
     Note: when no DRAFT_ANSWER_ tokens are present the draft term is 0 but
     cost_per_tool and missing_draft_penalty still apply (this is NOT a
     fallback to binary).

     Anti-collapse design (see tests/test_adc_reward.py):
     - cost_per_tool warms up linearly over adc_cost_warmup_steps so the
       tool-use skill can form before parsimony pressure is applied.
     - Budget-truncated rollouts (trajectory ends in a dangling tool call
       with no final answer, e.g. TRL rolled the tool result back) are NOT
       hit with the format penalty: the truncation was not a policy choice,
       and hammering it only punishes tool use. Pair with TRL's
       mask_truncated_completions so these rollouts are masked from the loss.
     - Format violations clamp the reward to min(r, 0) - format_penalty
       instead of subtracting final_bonus + draft_bonus, so the reward floor
       does not deepen with k (otherwise the worst rewards in the space are
       all tool-using trajectories and GRPO learns tool aversion).
     - IMPORTANT: this reward is designed for scale_rewards="none" (or
       "batch"). With per-group std normalization ("group"), all-correct
       groups have tiny std and the -cost_per_tool*k differences get
       amplified into full-size negative advantages on every solved prompt,
       which collapses tool use as accuracy rises.

     Incentive-compatibility notes (both exploits are tested in
     tests/test_adc_reward.py-style smoke checks):
     - An earlier version rewarded W→C transitions and penalized C→W. With
       bonus == penalty == c that sum telescopes to
       c*(1[final correct] - 1[first draft correct]), i.e. it PAID the policy
       to sandbag its first draft and then "correct" it.
     - A summed per-draft bonus with draft_bonus > cost_per_tool pays the
       policy to call unnecessary tools to farm correct drafts. Using the
       AVERAGE correctness of all answer statements bounds the bonus
       independently of trajectory length, closing that exploit too.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..utils.io import append_jsonl
from .prompt import count_unpaired_tool_turns, extract_answer_sequence, parse_final_answer


# ---------------------------------------------------------------------------
# CGC — Counterfactual Group Composition (Design A) support
# ---------------------------------------------------------------------------
# The harness disables tools for a fraction of each GRPO group (the "off arm")
# by answering every tool call with this sentinel message. The reward detects
# off-arm rollouts by the sentinel key and exempts them from tool costs and
# draft-format penalties. See DESIGN_A_CGC.md.

TOOLS_UNAVAILABLE_KEY = "tools_unavailable"
TOOLS_UNAVAILABLE_MSG = (
    '{"error": "tools_unavailable", '
    '"detail": "Tools are disabled for this attempt. Do not call any more tools. '
    'Answer directly now: end your reply with a single ANSWER_<TOKEN> line."}'
)


def _count_blocked_tool_msgs(completion: Any) -> int:
    if not isinstance(completion, list):
        return 0
    n = 0
    for msg in completion:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            if TOOLS_UNAVAILABLE_KEY in _msg_text(msg.get("content")):
                n += 1
    return n


def _transition_stats(y_hat_seq: List[Optional[str]], ground_truth: str) -> Tuple[int, int]:
    """(corrections, corruptions) = (#W→C, #C→W) over consecutive entries."""
    corrections = corruptions = 0
    for i in range(len(y_hat_seq) - 1):
        prev, curr = y_hat_seq[i], y_hat_seq[i + 1]
        if prev is None or curr is None:
            continue
        if prev != ground_truth and curr == ground_truth:
            corrections += 1
        elif prev == ground_truth and curr != ground_truth:
            corruptions += 1
    return corrections, corruptions


# ---------------------------------------------------------------------------
# Plaintext tool-call artefact detection
# ---------------------------------------------------------------------------

_TOOL_CALL_TAG_RE   = re.compile(r"<tool_call>", re.IGNORECASE)
_TOOLS_TAG_RE       = re.compile(r"<tools>", re.IGNORECASE)
_TOOL_CALLS_FIELD_RE = re.compile(r'"tool_calls"\s*:', re.IGNORECASE)
# Qwen3.5 renders native tool calls as <function=...>/<parameter=...> XML;
# those tags leaking into plain assistant text are artifacts too.
_TOOL_XML_TAG_RE    = re.compile(r"<function=|<parameter=", re.IGNORECASE)
_TOOL_NAMES = ("extractor_tool", "reasoner_tool", "verifier_tool")


def _has_plaintext_tool_artifacts(text: str) -> bool:
    if not text:
        return False
    if _TOOL_CALL_TAG_RE.search(text):
        return True
    if _TOOLS_TAG_RE.search(text):
        return True
    if _TOOL_CALLS_FIELD_RE.search(text):
        return True
    if _TOOL_XML_TAG_RE.search(text):
        return True
    for name in _TOOL_NAMES:
        if re.search(rf"\b{re.escape(name)}\s*[\(\{{:]", text, flags=re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Completion parsing helpers
# ---------------------------------------------------------------------------

def _msg_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for blk in content:
            if isinstance(blk, dict) and "text" in blk:
                out.append(str(blk.get("text", "")))
        return "\n".join(out)
    return str(content)


def _extract_completion_stats(completion: Any) -> Dict[str, Any]:
    """Pull routing stats from a completion (TRL message-list format)."""
    if not isinstance(completion, list):
        text = _msg_text(completion)
        return {
            "last_assistant_text": text,
            "tool_calls": 0,
            "tool_msgs": 0,
            "tool_names_called": [],
            "last_msg_has_tool_calls": False,
            "last_msg_has_plaintext_artifacts": _has_plaintext_tool_artifacts(text),
        }

    assistant_msgs = [m for m in completion if isinstance(m, dict) and m.get("role") == "assistant"]
    tool_msgs      = [m for m in completion if isinstance(m, dict) and m.get("role") == "tool"]

    tool_calls = 0
    tool_names_called: List[str] = []
    for m in assistant_msgs:
        tc = m.get("tool_calls")
        if isinstance(tc, list):
            tool_calls += len(tc)
            for entry in tc:
                fn = (entry.get("function", {}) or {}).get("name", "") if isinstance(entry, dict) else ""
                if fn:
                    tool_names_called.append(str(fn))

    last_text = ""
    last_has_tc = False
    if assistant_msgs:
        last_text   = _msg_text(assistant_msgs[-1].get("content"))
        last_has_tc = bool(assistant_msgs[-1].get("tool_calls"))

    return {
        "last_assistant_text": last_text,
        "tool_calls": tool_calls,
        "tool_msgs": len(tool_msgs),
        "tool_names_called": tool_names_called,
        "last_msg_has_tool_calls": last_has_tc,
        "last_msg_has_plaintext_artifacts": _has_plaintext_tool_artifacts(last_text),
    }


def _ensure_list(x: Any, n: int) -> List[Any]:
    if isinstance(x, list):
        if len(x) == n:
            return x
        if not x:
            return [None] * n
        return (x * ((n // len(x)) + 1))[:n]
    return [x] * n


# ---------------------------------------------------------------------------
# CCR helpers (legacy)
# ---------------------------------------------------------------------------

def _ccr_implicit_confidence(k: int, k_max: int, p_high: float, p_low: float) -> float:
    if k_max <= 0:
        return max(1e-7, min(1 - 1e-7, p_high))
    t = min(1.0, max(0.0, k / k_max))
    p = p_high + (p_low - p_high) * t
    return max(1e-7, min(1 - 1e-7, p))


def _ccr_log_reward(correct: bool, k: int, k_max: int, p_high: float, p_low: float) -> float:
    p = _ccr_implicit_confidence(k, k_max, p_high, p_low)
    return math.log(p) if correct else math.log(1.0 - p)


# ---------------------------------------------------------------------------
# ADC helpers
# ---------------------------------------------------------------------------

def _compute_adc_reward(
    y_hat_seq: List[Optional[str]],
    ground_truth: str,
    n_tools: int,
    draft_bonus: float,
    missing_draft_penalty: float,
    final_bonus: float,
    cost_per_tool: float,
    has_final: bool,
    variant: str = "anytime",
) -> Tuple[float, Dict[str, Any]]:
    """Compute ADC anytime reward from the answer sequence.

    y_hat_seq: candidate answers in chronological order (DRAFT_ANSWER_ tokens,
               then the final ANSWER_ if one was parsed), from the completion.
    has_final: whether the last element of y_hat_seq is a parsed final answer.
    Returns (total_reward, stats_dict).

    Reward = draft_bonus * (fraction of ALL answer statements that are correct)
             - missing_draft_penalty * max(0, n_tools - n_drafts)
             + final_bonus * 1[final correct]
             - cost_per_tool * n_tools

    The draft bonus is an AVERAGE, not a sum: it is bounded by draft_bonus no
    matter how many tools are called, so extra tool calls cannot farm it (a
    summed bonus with draft_bonus > cost_per_tool would pay the policy to call
    unnecessary tools). A deliberately wrong draft strictly lowers the average,
    so honest best-guess drafts remain the unique optimal draft policy.

    variant selects the process-reward term. "anytime" is the fixed design;
    "transition" and "sum" reproduce the two provably exploitable designs and
    exist ONLY as ablation arms for the incentive-compatibility experiment:
      - "transition": +draft_bonus per W→C, -draft_bonus per C→W (telescopes to
        paying for a deliberately wrong first draft — sandbagging incentive);
      - "sum": +draft_bonus per correct draft (farmable by superfluous tool
        calls whenever draft_bonus > cost_per_tool).
    """
    if has_final and y_hat_seq:
        drafts = y_hat_seq[:-1]
        final_pred = y_hat_seq[-1]
    else:
        drafts = list(y_hat_seq)
        final_pred = None

    entries = [y for y in y_hat_seq if y is not None]
    n_correct_drafts = sum(1 for d in drafts if d is not None and d == ground_truth)
    n_correct_entries = sum(1 for d in entries if d == ground_truth)

    missing_drafts = max(0, int(n_tools) - len(drafts))

    final_ok = (final_pred == ground_truth) if final_pred is not None else False
    f_reward = final_bonus if final_ok else 0.0

    t_cost = n_tools * cost_per_tool

    # W→C / C→W transitions: diagnostics for "anytime"/"sum", reward for "transition".
    corrections = 0
    corruptions = 0
    no_change = 0
    for i in range(len(y_hat_seq) - 1):
        prev = y_hat_seq[i]
        curr = y_hat_seq[i + 1]
        if prev is None or curr is None:
            continue
        prev_ok = (prev == ground_truth)
        curr_ok = (curr == ground_truth)
        if not prev_ok and curr_ok:
            corrections += 1
        elif prev_ok and not curr_ok:
            corruptions += 1
        else:
            no_change += 1

    if variant == "transition":
        draft_reward = draft_bonus * corrections - draft_bonus * corruptions
        miss_penalty = 0.0
    elif variant == "sum":
        draft_reward = draft_bonus * n_correct_drafts
        miss_penalty = missing_draft_penalty * missing_drafts
    else:  # "anytime"
        draft_reward = draft_bonus * (n_correct_entries / len(entries)) if entries else 0.0
        miss_penalty = missing_draft_penalty * missing_drafts

    total = f_reward + draft_reward - miss_penalty - t_cost
    return total, {
        "variant": variant,
        "corrections": corrections,
        "corruptions": corruptions,
        "no_change": no_change,
        "n_drafts": len(drafts),
        "n_correct_drafts": n_correct_drafts,
        "missing_drafts": missing_drafts,
        "draft_reward": round(draft_reward, 4),
        "missing_draft_penalty": round(miss_penalty, 4),
        "final_reward": round(f_reward, 4),
        "tool_cost": round(t_cost, 4),
        "y_hat_seq": [str(y) if y is not None else None for y in y_hat_seq],
        "final_correct": bool(final_ok),
    }


# ---------------------------------------------------------------------------
# Post-hoc analysis utilities
# ---------------------------------------------------------------------------

def compute_ece(
    records: List[Dict[str, Any]],
    k_max: int = 3,
    p_high: float = 0.9,
    p_low: float = 0.2,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error from trace records."""
    buckets: Dict[int, List[bool]] = defaultdict(list)
    for rec in records:
        k       = int(rec.get("tool_calls", 0))
        correct = bool(rec.get("correct", rec.get("reward", 0) > 0))
        buckets[k].append(correct)

    n_total = sum(len(v) for v in buckets.values())
    if n_total == 0:
        return {"ece": 0.0, "buckets": {}, "n_total": 0}

    ece = 0.0
    bucket_stats: Dict[int, Any] = {}
    for k in range(k_max + 1):
        items = buckets.get(k, [])
        if not items:
            bucket_stats[k] = None
            continue
        acc    = sum(items) / len(items)
        p      = _ccr_implicit_confidence(k, k_max, p_high, p_low)
        weight = len(items) / n_total
        ece   += weight * abs(p - acc)
        bucket_stats[k] = {
            "n": len(items),
            "accuracy": round(acc, 4),
            "implicit_confidence": round(p, 4),
            "calibration_gap": round(abs(p - acc), 4),
            "weight": round(weight, 4),
        }

    return {"ece": round(ece, 6), "buckets": bucket_stats, "n_total": n_total}


def compute_routing_entropy(records: List[Dict[str, Any]], k_max: int = 3) -> Dict[str, Any]:
    """Compute routing entropy H over the empirical tool-call distribution."""
    counts: Dict[int, int] = defaultdict(int)
    for rec in records:
        k = min(int(rec.get("tool_calls", 0)), k_max)
        counts[k] += 1

    n_total = sum(counts.values())
    if n_total == 0:
        return {"entropy": 0.0, "distribution": {}, "n_total": 0}

    entropy = 0.0
    distribution: Dict[str, float] = {}
    for k in range(k_max + 1):
        n_k  = counts.get(k, 0)
        frac = n_k / n_total
        distribution[str(k)] = round(frac, 4)
        if frac > 0:
            entropy -= frac * math.log(frac)

    return {
        "entropy": round(entropy, 6),
        "max_entropy": round(math.log(k_max + 1), 6),
        "normalized_entropy": round(entropy / math.log(k_max + 1), 4) if k_max > 0 else 1.0,
        "distribution": distribution,
        "n_total": n_total,
    }


def compute_risk_coverage(records: List[Dict[str, Any]], k_max: int = 3) -> Dict[str, Any]:
    """Selective-prediction view of routing: treat fewer tool calls as higher
    confidence and compute the risk-coverage curve + AURC.

    At coverage level for confidence threshold t, the manager "accepts" all
    examples answered with k <= t tools; risk is the error rate among accepted.
    This avoids imputing a synthetic confidence value (unlike compute_ece).
    """
    by_k: Dict[int, List[bool]] = defaultdict(list)
    for rec in records:
        k = min(int(rec.get("tool_calls", 0)), k_max)
        correct = bool(rec.get("correct", rec.get("reward", 0) > 0))
        by_k[k].append(correct)

    n_total = sum(len(v) for v in by_k.values())
    if n_total == 0:
        return {"aurc": 0.0, "curve": [], "n_total": 0}

    curve: List[Dict[str, float]] = []
    acc_n = 0
    acc_correct = 0
    for k in range(k_max + 1):
        items = by_k.get(k, [])
        acc_n += len(items)
        acc_correct += sum(items)
        if acc_n == 0:
            continue
        coverage = acc_n / n_total
        risk = 1.0 - acc_correct / acc_n
        curve.append({"k_threshold": k, "coverage": round(coverage, 4), "risk": round(risk, 4)})

    # AURC via trapezoid over the coverage axis (prepend coverage=0 at first risk).
    aurc = 0.0
    prev_cov, prev_risk = 0.0, (curve[0]["risk"] if curve else 0.0)
    for pt in curve:
        aurc += (pt["coverage"] - prev_cov) * (pt["risk"] + prev_risk) / 2.0
        prev_cov, prev_risk = pt["coverage"], pt["risk"]

    return {"aurc": round(aurc, 6), "curve": curve, "n_total": n_total}


def compute_deliberation_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute ADC-specific statistics: correction/corruption rates per k."""
    total = len(records)
    if total == 0:
        return {}

    corrections = sum(r.get("corrections", 0) for r in records)
    corruptions = sum(r.get("corruptions", 0) for r in records)
    has_drafts  = sum(1 for r in records if r.get("n_drafts", 0) > 0)
    stop_correct = sum(1 for r in records if r.get("correct") and r.get("tool_calls", 0) == 0)

    return {
        "total": total,
        "correction_rate": round(corrections / total, 4),
        "corruption_rate": round(corruptions / total, 4),
        "net_gain": round((corrections - corruptions) / total, 4),
        "pct_with_draft_answers": round(has_drafts / total, 4),
        "pct_stopped_immediately_correct": round(stop_correct / total, 4),
    }


# ---------------------------------------------------------------------------
# Reward function builder
# ---------------------------------------------------------------------------

def build_reward_funcs(
    fail_buffer_jsonl: Optional[str] = None,
    raw_trace_jsonl: Optional[str] = None,
    routing_efficiency_bonus: float = 0.0,
    tool_use_bonus: float = 0.0,
    ccr_mode: bool = False,
    ccr_p_high: float = 0.9,
    ccr_p_low: float = 0.6,
    ccr_k_max: int = 3,
    adc_mode: bool = False,
    adc_cost_per_tool: float = 0.02,
    adc_draft_bonus: float = 0.02,
    adc_missing_draft_penalty: float = 0.1,
    adc_final_bonus: float = 1.0,
    adc_variant: str = "anytime",
    adc_format_penalty: float = 0.2,
    adc_cost_warmup_steps: int = 100,
    cgc_mode: bool = False,
    cgc_cost_per_tool: float = 0.01,
    cgc_missing_draft_penalty: float = 0.05,
    cgc_cost_warmup_steps: int = 100,
    cgc_flatten: str = "novar",
    is_main_process: bool = True,
):
    """Construct the reward function list passed to GRPOTrainer.

    Mode priority: cgc_mode > adc_mode > ccr_mode > binary.

    CGC mode (Design A — counterfactual group composition):
        Plain binary correctness plus a small per-tool cost, paired with a
        harness that disables tools for half of each GRPO group (off arm,
        detected here via the tools_unavailable sentinel). The routing signal
        comes from group COMPOSITION (the group mean contains the no-tool
        counterfactual), not from reward shaping. Off-arm rollouts are exempt
        from tool cost and draft-format penalties. Drafts are pure telemetry:
        no draft bonus, only a small per-tool-TURN missing-draft penalty on
        the on arm (per-turn pairing closes the post-hoc draft hole, see
        ADC_RESIDUAL_HOLES.md §2). cgc_flatten="novar" zeroes the gradient of
        groups with no correctness variance (all right / all wrong) by
        setting every reward in the group to the group mean — soft dynamic
        sampling that removes the pure-cost anti-tool drip in hopeless or
        trivial groups (the LLD collapse fuel) while mixed groups keep cost
        as a parsimony tiebreaker.

    ADC mode (recommended):
        Anytime per-draft correctness reward + final correctness - tool cost.
        Incentive-compatible: honest drafts are optimal (see module docstring).
        adc_cost_warmup_steps > 0 linearly ramps cost_per_tool from 0 to its
        target over that many steps (0 disables the warmup). Calibrate the
        target from traces: cost_per_tool should be at most ~1/3-1/2 of
        (corrections - corruptions) / total tool calls * final_bonus, so
        tools stay net-positive in expectation wherever they actually help.

    CCR mode (legacy):
        Log scoring rule on implicit confidence from k.
        Requires p_low > 0.5 to avoid reward inversion at k>=2.
        Default changed to p_low=0.6 (was 0.2, which caused corruption).

    Binary mode:
        R = 1.0 if correct else 0.0, plus optional bonuses.
    """

    # Fallback step counter for the cost warmup when TRL does not pass
    # trainer_state to reward functions (counts reward invocations, which
    # tracks optimizer steps closely enough for a warmup schedule).
    _call_count = {"n": 0}

    def reward_fn(
        prompts=None,
        completions=None,
        ground_truth=None,
        example_id=None,
        choice_keys=None,
        **kwargs,
    ) -> List[float]:
        _call_count["n"] += 1
        _ts = kwargs.get("trainer_state")
        _step = getattr(_ts, "global_step", None) if _ts is not None else None
        if _step is None:
            _step = _call_count["n"]

        def _warm(target: float, warmup_steps: int) -> float:
            if warmup_steps > 0:
                return target * min(1.0, float(_step) / float(warmup_steps))
            return target

        cost_now = _warm(adc_cost_per_tool, adc_cost_warmup_steps)
        cgc_cost_now = _warm(cgc_cost_per_tool, cgc_cost_warmup_steps)

        n     = len(completions)
        gts   = _ensure_list(ground_truth, n)
        eids  = _ensure_list(example_id, n)
        ck_lists = _ensure_list(choice_keys, n)

        rewards: List[float] = []
        corrects: List[bool] = []
        fail_rows:  List[Dict[str, Any]] = []
        trace_rows: List[Dict[str, Any]] = []
        trace_index: List[int] = []  # completion index of each trace row

        for _ci, (c, gt, eid, keys) in enumerate(zip(completions, gts, eids, ck_lists)):
            stats    = _extract_completion_stats(c)
            keys_list = list(keys) if isinstance(keys, (list, tuple)) else []
            pred     = parse_final_answer(stats["last_assistant_text"], keys_list)

            valid_format  = pred is not None
            # 刷屏防护:干净的 final turn 只应有一个 ANSWER_ 且不超长。GRPO 会漂移到
            # 重复刷 ANSWER_ 填满 budget(每个都含正确答案骗过 parse、又不打 <|im_end|>),
            # tool_rate 随之崩。多个/零个 ANSWER_ 或超长 => 判定 format 失败。
            # (?<!DRAFT_): SFT 学出的"final turn 先 draft 再 answer"习惯不该吃格式罚,
            # 否则 RL 初期定点爆破的正是和工具使用绑定的 deliberation 行为。
            _ans_hits = len(re.findall(r"(?<!DRAFT_)ANSWER_[A-Za-z0-9]", stats["last_assistant_text"]))
            if _ans_hits != 1 or len(stats["last_assistant_text"]) > 300:
                valid_format = False
            no_artifacts  = not stats["last_msg_has_plaintext_artifacts"]
            no_tc_in_final = not stats["last_msg_has_tool_calls"]
            base_correct  = bool(valid_format and no_artifacts and no_tc_in_final and pred == gt)
            k = int(stats["tool_calls"])

            # ---- CGC mode (Design A: paired counterfactual arms) ----
            cgc_stats: Dict[str, Any] = {}
            if cgc_mode:
                n_blocked = _count_blocked_tool_msgs(c)
                k_eff = max(0, int(stats["tool_msgs"]) - n_blocked)
                is_truncated = bool(stats["last_msg_has_tool_calls"]) and pred is None
                n_tool_turns, n_unpaired = count_unpaired_tool_turns(c, keys_list)
                # Off arm (attempted a call, got the sentinel): pure binary,
                # no cost, no draft-format penalty — the attempt was not a
                # policy failure, the harness said no.
                miss = 0.0 if n_blocked > 0 else cgc_missing_draft_penalty * n_unpaired
                cgc_r = (1.0 if base_correct else 0.0) - cgc_cost_now * k_eff - miss
                if not (valid_format and no_artifacts and no_tc_in_final) and not is_truncated:
                    cgc_r = min(cgc_r, 0.0) - adc_format_penalty
                reward = float(cgc_r)

                # Draft telemetry (out of the reward entirely — exogenous
                # measurement channel; see DESIGN_A_CGC.md).
                y_seq = extract_answer_sequence(c, keys_list)
                drafts_seq = y_seq[:-1] if (pred is not None and y_seq) else list(y_seq)
                first_draft = next((d for d in drafts_seq if d is not None), None)
                n_corr, n_corrupt = _transition_stats(y_seq, str(gt))
                cgc_stats = {
                    "k_eff": int(k_eff),
                    "blocked_tool_calls": int(n_blocked),
                    "off_arm": bool(n_blocked > 0),
                    "used_tools": bool(k_eff > 0),
                    "n_tool_turns": int(n_tool_turns),
                    "n_unpaired_tool_turns": int(n_unpaired),
                    "truncated": bool(is_truncated),
                    "y_hat_seq": [str(y) if y is not None else None for y in y_seq],
                    "first_draft": first_draft,
                    "first_draft_correct": (bool(first_draft == gt) if first_draft is not None else None),
                    "corrections": int(n_corr),
                    "corruptions": int(n_corrupt),
                    "cost_per_tool_effective": round(float(cgc_cost_now), 4),
                }

            # ---- ADC mode ----
            elif adc_mode:
                y_hat_seq = extract_answer_sequence(c, keys_list)
                # Dangling tool call with no final answer = the rollout was cut
                # by the completion budget (TRL rolls the tool result back),
                # not a policy choice. Exempt it from the format penalty —
                # otherwise every budget event is a large anti-tool gradient
                # that only tool-using trajectories can receive.
                is_truncated = bool(stats["last_msg_has_tool_calls"]) and pred is None
                adc_r, adc_stats = _compute_adc_reward(
                    y_hat_seq=y_hat_seq,
                    ground_truth=str(gt),
                    n_tools=k,
                    draft_bonus=adc_draft_bonus,
                    missing_draft_penalty=adc_missing_draft_penalty,
                    final_bonus=adc_final_bonus,
                    cost_per_tool=cost_now,
                    has_final=(pred is not None),
                    variant=adc_variant,
                )
                if not (valid_format and no_artifacts and no_tc_in_final) and not is_truncated:
                    # Format violation chosen by the policy: forfeit any
                    # positive reward and pay a flat penalty. Clamp instead of
                    # subtracting final_bonus+draft_bonus so the floor does not
                    # deepen with k (the old blanket -1.2 put every deep reward
                    # valley on the tool-using side of the space).
                    adc_r = min(adc_r, 0.0) - adc_format_penalty
                adc_stats["truncated"] = bool(is_truncated)
                reward = float(adc_r)

            # ---- CCR mode (legacy) ----
            elif ccr_mode:
                reward = _ccr_log_reward(base_correct, k, ccr_k_max, ccr_p_high, ccr_p_low)
                if not (valid_format and no_artifacts and no_tc_in_final):
                    reward = math.log(1.0 - ccr_p_high + 1e-7)

            # ---- Binary mode ----
            else:
                reward = 1.0 if base_correct else 0.0
                if base_correct and routing_efficiency_bonus > 0.0:
                    saved  = max(0, ccr_k_max - k)
                    reward = reward + routing_efficiency_bonus * saved
                if base_correct and tool_use_bonus > 0.0 and k > 0:
                    reward = reward + tool_use_bonus

            rewards.append(float(reward))
            corrects.append(bool(base_correct))

            if not base_correct and is_main_process and fail_buffer_jsonl:
                fail_rows.append({
                    "ts": int(time.time()),
                    "example_id": int(eid) if eid is not None else None,
                    "ground_truth": gt,
                    "pred": pred,
                    "valid_format": bool(valid_format),
                    "no_artifacts": bool(no_artifacts),
                    "no_tc_in_final": bool(no_tc_in_final),
                    "tool_calls": k,
                    "tool_msgs": int(stats["tool_msgs"]),
                    "tool_names_called": list(stats["tool_names_called"]),
                    "last_assistant_text": stats["last_assistant_text"][:2000],
                })

            if is_main_process and raw_trace_jsonl:
                trace_entry: Dict[str, Any] = {
                    "ts": int(time.time()),
                    "example_id": int(eid) if eid is not None else None,
                    "ground_truth": gt,
                    "pred": pred,
                    "correct": bool(base_correct),
                    "reward": float(reward),
                    "tool_calls": k,
                    "tool_msgs": int(stats["tool_msgs"]),
                    "tool_names_called": list(stats["tool_names_called"]),
                    "reward_mode": (
                        "cgc" if cgc_mode else (
                            f"adc:{adc_variant}" if adc_mode else ("ccr" if ccr_mode else "binary")
                        )
                    ),
                }
                if cgc_mode:
                    trace_entry.update(cgc_stats)
                elif adc_mode:
                    trace_entry.update({
                        "y_hat_seq": adc_stats.get("y_hat_seq", []),
                        "corrections": adc_stats.get("corrections", 0),
                        "corruptions": adc_stats.get("corruptions", 0),
                        "n_drafts": adc_stats.get("n_drafts", 0),
                        "n_correct_drafts": adc_stats.get("n_correct_drafts", 0),
                        "missing_drafts": adc_stats.get("missing_drafts", 0),
                        "draft_reward": adc_stats.get("draft_reward", 0.0),
                        "final_reward": adc_stats.get("final_reward", 0.0),
                        "tool_cost": adc_stats.get("tool_cost", 0.0),
                        "truncated": adc_stats.get("truncated", False),
                        "cost_per_tool_effective": round(float(cost_now), 4),
                    })
                elif ccr_mode:
                    trace_entry["implicit_confidence"] = round(
                        _ccr_implicit_confidence(k, ccr_k_max, ccr_p_high, ccr_p_low), 4
                    )
                trace_rows.append(trace_entry)
                trace_index.append(_ci)

        # ---- CGC group flattening (soft dynamic sampling) ----
        # Groups with zero correctness variance (all right / all wrong) carry
        # no routing signal; their only within-group reward differences are
        # cost/penalty terms, which are all tool-sided and act as a constant
        # anti-tool drip (the collapse fuel on hard datasets). Setting every
        # reward in such a group to the group mean makes all advantages 0.
        # Groups are identified by example_id (robust to batch ordering).
        if cgc_mode and cgc_flatten == "novar":
            _groups: Dict[Any, List[int]] = defaultdict(list)
            for _i, _e in enumerate(eids):
                if _e is not None:
                    _groups[_e].append(_i)
            _flattened: set = set()
            for _e, _idxs in _groups.items():
                if len(_idxs) < 2:
                    continue
                if len({corrects[_i] for _i in _idxs}) == 1:
                    _m = sum(rewards[_i] for _i in _idxs) / len(_idxs)
                    for _i in _idxs:
                        rewards[_i] = float(_m)
                        _flattened.add(_i)
            for _pos, _i in enumerate(trace_index):
                trace_rows[_pos]["reward"] = float(rewards[_i])
                trace_rows[_pos]["flattened"] = bool(_i in _flattened)

        if fail_rows and fail_buffer_jsonl:
            append_jsonl(fail_buffer_jsonl, fail_rows)
        if trace_rows and raw_trace_jsonl:
            append_jsonl(raw_trace_jsonl, trace_rows)

        return rewards

    if cgc_mode:
        reward_fn.__name__ = "cgc_paired_binary"
    elif adc_mode:
        reward_fn.__name__ = f"adc_{adc_variant}"
    elif ccr_mode:
        reward_fn.__name__ = "ccr_log_scoring"
    else:
        reward_fn.__name__ = "binary_outcome_with_format"
    return [reward_fn]


# ---------------------------------------------------------------------------
# Convenience bare-function export
# ---------------------------------------------------------------------------

def binary_outcome_reward(
    prompts=None,
    completions=None,
    ground_truth=None,
    example_id=None,
    choice_keys=None,
    **kwargs,
) -> List[float]:
    fn_list = build_reward_funcs()
    return fn_list[0](
        prompts=prompts,
        completions=completions,
        ground_truth=ground_truth,
        example_id=example_id,
        choice_keys=choice_keys,
        **kwargs,
    )
