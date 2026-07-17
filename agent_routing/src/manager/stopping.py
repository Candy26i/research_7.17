"""Harness-level optimal stopping for the manager agent ("stop probe").

Motivation (first principles)
-----------------------------
The manager cannot learn WHEN to stop calling tools through reward shaping
because (a) a single rollout never observes the counterfactual "what if I had
stopped at k-1", (b) once the policy collapses to always-k_max there is no
variance in k inside a GRPO group and therefore no gradient about stopping,
and (c) a scalar reward cannot robustly encode the lexicographic preference
"accuracy first, then fewer tools".

This module therefore moves the stopping decision OUT of the policy weights
and INTO the harness:

  1. `collect_counterfactuals` (in pipeline/stages.py) force-continues the
     manager to k_max tools and probes a final answer at EVERY stage
     k = 0..k_max, so the per-question marginal-benefit curve
     Delta(k) = correct(k+1) - correct(k) becomes directly measurable.
  2. `fit_stop_probe` fits a tiny calibrated logistic model
     P(current answer is correct | observable stage features) on those
     counterfactual records. Features are label-free at inference time
     (self-consistency vote agreement, answer stability, k, lengths, ...).
  3. `threshold_sweep` turns the probe into a family of stopping policies
     "stop at the first stage where P >= theta" and traces the
     accuracy-vs-cost Pareto curve; `choose_threshold` picks the operating
     point that minimizes tool calls subject to an accuracy constraint.
  4. `build_distill_rows` optionally converts probe-selected minimal correct
     trajectories into manager SFT rows (same per-turn format as
     manager/evolve.py) so the stopping behavior can be internalized without
     touching the RL algorithm or the reward function.

Everything here is deliberately free of torch / transformers imports so it can
be unit-tested on CPU-only machines.
"""
from __future__ import annotations

import json
import math
import os
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .prompt import _label_to_token


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical stage/step budget (0..K_MAX_DEFAULT tools).
K_MAX_DEFAULT = 3

#: User-role message appended on the PROBE BRANCH ONLY (never enters the main
#: trajectory) to force the manager to commit to an answer at the current stage.
STOP_PROBE_USER_MSG = (
    "No further tools are available for this question. Based on everything so "
    "far, provide your final answer now. Give at most two sentences of "
    "reasoning, then end with exactly one line: ANSWER_<CHOICE>."
)

#: Ordered feature names. `stage_features` MUST return floats in this order,
#: and a saved probe refuses to score a mismatched feature list.
FEATURE_NAMES: List[str] = [
    "k_norm",                  # k / k_max
    "is_k0",                   # 1.0 if no tool called yet
    "agreement_top",           # fraction of valid votes on the majority answer
    "greedy_matches_majority", # 1.0 if greedy answer == vote majority
    "valid_vote_frac",         # parsed votes / total votes
    "vote_entropy_norm",       # entropy of vote distribution / log(n_choices)
    "stable_from_prev",        # 1.0 if greedy answer unchanged vs previous stage
    "chance_rate",             # 1 / n_choices
    "log_qlen",                # log1p(len(question)) / 10
    "log_ctxlen",              # log1p(len(context)) / 10
]


# ---------------------------------------------------------------------------
# Vote statistics and feature extraction
# ---------------------------------------------------------------------------

def vote_stats(votes: Sequence[Optional[str]], n_choices: int) -> Dict[str, Any]:
    """Aggregate self-consistency votes into calibration-relevant statistics.

    `votes` may contain None entries (unparseable samples). Ties on the
    majority are broken deterministically by (count desc, key asc).
    """
    n_total = len(votes)
    valid = [v for v in votes if v is not None and str(v) != ""]
    if not valid:
        return {
            "majority": None,
            "agreement_top": 0.0,
            "valid_vote_frac": 0.0,
            "vote_entropy_norm": 1.0,
        }
    counts = Counter(str(v) for v in valid)
    majority, top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    agreement = top / len(valid)
    probs = [c / len(valid) for c in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    denom = math.log(max(2, int(n_choices)))
    return {
        "majority": majority,
        "agreement_top": agreement,
        "valid_vote_frac": len(valid) / max(1, n_total),
        "vote_entropy_norm": min(1.0, entropy / denom),
    }


def stage_features(
    k: int,
    k_max: int,
    greedy_pred: Optional[str],
    votes: Sequence[Optional[str]],
    prev_pred: Optional[str],
    n_choices: int,
    question_len: int,
    context_len: int,
) -> List[float]:
    """Build the label-free feature vector for one deliberation stage.

    All inputs are observable by the harness at inference time; the ground
    truth never enters here.
    """
    vs = vote_stats(votes, n_choices)
    greedy_matches_majority = float(
        greedy_pred is not None
        and vs["majority"] is not None
        and str(greedy_pred) == str(vs["majority"])
    )
    stable = float(
        k > 0
        and greedy_pred is not None
        and prev_pred is not None
        and str(greedy_pred) == str(prev_pred)
    )
    k_max_eff = max(1, int(k_max))
    return [
        float(k) / k_max_eff,
        1.0 if k == 0 else 0.0,
        float(vs["agreement_top"]),
        greedy_matches_majority,
        float(vs["valid_vote_frac"]),
        float(vs["vote_entropy_norm"]),
        stable,
        1.0 / max(1, int(n_choices)),
        math.log1p(max(0, int(question_len))) / 10.0,
        math.log1p(max(0, int(context_len))) / 10.0,
    ]


def record_stage_feature_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand one counterfactual record into per-stage (features, label) rows.

    A record is one JSONL row written by run_collect_counterfactuals with at
    least: ground_truth, n_choices, question_len, context_len, k_max, stages
    (each stage: k, pred, votes).
    """
    gt = str(record.get("ground_truth"))
    n_choices = int(record.get("n_choices") or 1)
    qlen = int(record.get("question_len") or 0)
    clen = int(record.get("context_len") or 0)
    k_max = int(record.get("k_max") or K_MAX_DEFAULT)
    stages = record.get("stages") or []
    rows: List[Dict[str, Any]] = []
    prev_pred: Optional[str] = None
    for st in stages:
        k = int(st.get("k") or 0)
        pred = st.get("pred")
        feats = stage_features(
            k=k, k_max=k_max, greedy_pred=pred, votes=st.get("votes") or [],
            prev_pred=prev_pred, n_choices=n_choices,
            question_len=qlen, context_len=clen,
        )
        rows.append({
            "example_id": record.get("example_id"),
            "k": k,
            "features": feats,
            "label": 1.0 if (pred is not None and str(pred) == gt) else 0.0,
        })
        prev_pred = pred
    return rows


# ---------------------------------------------------------------------------
# Logistic probe
# ---------------------------------------------------------------------------

def _sigmoid(z):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class StopProbe:
    """Calibrated logistic model P(current greedy answer is correct | stage).

    Stores its own feature standardization so scoring is self-contained.
    `threshold` is the recommended operating point chosen by choose_threshold;
    callers may override it at deployment time.
    """
    feature_names: List[str]
    mean: List[float]
    std: List[float]
    weights: List[float]
    bias: float
    threshold: float = 0.5
    meta: Dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, features: Sequence[float]) -> float:
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"StopProbe expects {len(self.feature_names)} features "
                f"({self.feature_names}), got {len(features)}"
            )
        z = self.bias
        for f, m, s, w in zip(features, self.mean, self.std, self.weights):
            s_eff = s if s > 1e-12 else 1.0
            z += w * ((float(f) - m) / s_eff)
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": [float(x) for x in self.mean],
            "std": [float(x) for x in self.std],
            "weights": [float(x) for x in self.weights],
            "bias": float(self.bias),
            "threshold": float(self.threshold),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StopProbe":
        names = list(d["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError(
                "Stop probe feature schema mismatch: probe was fit with "
                f"{names}, this code expects {FEATURE_NAMES}. Re-fit the probe."
            )
        return cls(
            feature_names=names,
            mean=[float(x) for x in d["mean"]],
            std=[float(x) for x in d["std"]],
            weights=[float(x) for x in d["weights"]],
            bias=float(d["bias"]),
            threshold=float(d.get("threshold", 0.5)),
            meta=dict(d.get("meta") or {}),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "StopProbe":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def fit_logistic(
    X, y, l2: float = 1e-3, lr: float = 0.5, iters: int = 4000,
) -> Tuple[Any, float]:
    """Full-batch gradient-descent L2 logistic regression on standardized X.

    Deterministic (zero init, fixed schedule). Returns (weights, bias).
    """
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(int(iters)):
        p = _sigmoid(X @ w + b)
        g = p - y
        gw = X.T @ g / n + l2 * w
        gb = float(g.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def fit_stop_probe(
    records: List[Dict[str, Any]],
    l2: float = 1e-3,
    lr: float = 0.5,
    iters: int = 4000,
) -> Tuple[StopProbe, Dict[str, Any]]:
    """Fit the stop probe on counterfactual records. Returns (probe, metrics)."""
    import numpy as np
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.extend(record_stage_feature_rows(rec))
    if len(rows) < 10:
        raise ValueError(f"Too few probe training rows: {len(rows)} (need >= 10)")

    X = np.asarray([r["features"] for r in rows], dtype=np.float64)
    y = np.asarray([r["label"] for r in rows], dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_eff = np.where(std > 1e-12, std, 1.0)
    Xs = (X - mean) / std_eff

    w, b = fit_logistic(Xs, y, l2=l2, lr=lr, iters=iters)
    p = _sigmoid(Xs @ w + b)

    probe = StopProbe(
        feature_names=list(FEATURE_NAMES),
        mean=[float(x) for x in mean],
        std=[float(x) for x in std_eff],
        weights=[float(x) for x in w],
        bias=float(b),
        meta={
            "n_records": len(records),
            "n_stage_rows": len(rows),
            "l2": l2,
            "base_rate": float(y.mean()),
        },
    )
    metrics = {
        "n_stage_rows": len(rows),
        "base_rate": round(float(y.mean()), 4),
        "auc": rank_auc(y, p),
        "brier": round(float(((p - y) ** 2).mean()), 4),
        "ece": expected_calibration_error(y, p),
        "weights": {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, w)},
        "bias": round(float(b), 4),
    }
    return probe, metrics


def rank_auc(y, p) -> Optional[float]:
    """Tie-aware rank AUC (Mann-Whitney). None if only one class present."""
    import numpy as np
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    # average ranks over ties
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = float(ranks[y == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(auc, 4)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    import numpy as np
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return round(float(ece), 4)


# ---------------------------------------------------------------------------
# Stopping-policy simulation on counterfactual records
# ---------------------------------------------------------------------------

def example_stage_probs(record: Dict[str, Any], probe: StopProbe) -> List[float]:
    """Probe probabilities for each stage of one record, in stage order."""
    return [probe.predict_proba(r["features"]) for r in record_stage_feature_rows(record)]


def _stage_correct(record: Dict[str, Any], idx: int) -> bool:
    st = (record.get("stages") or [])[idx]
    pred = st.get("pred")
    return pred is not None and str(pred) == str(record.get("ground_truth"))


def simulate_threshold(
    records: List[Dict[str, Any]],
    probs_list: List[List[float]],
    threshold: float,
) -> Dict[str, Any]:
    """Simulate 'stop at first stage with P >= threshold, else last stage'."""
    n = 0
    n_correct = 0
    k_sum = 0
    k_dist: Dict[str, int] = {}
    for rec, probs in zip(records, probs_list):
        stages = rec.get("stages") or []
        if not stages:
            continue
        stop_idx = len(stages) - 1
        for i, p in enumerate(probs):
            if p >= threshold:
                stop_idx = i
                break
        n += 1
        k_sum += stop_idx
        k_dist[str(stop_idx)] = k_dist.get(str(stop_idx), 0) + 1
        if _stage_correct(rec, stop_idx):
            n_correct += 1
    return {
        "threshold": round(float(threshold), 4),
        "n": n,
        "accuracy": round(n_correct / max(1, n), 4),
        "avg_tool_calls": round(k_sum / max(1, n), 4),
        "k_dist": dict(sorted(k_dist.items())),
    }


def threshold_sweep(
    records: List[Dict[str, Any]],
    probe: StopProbe,
    grid: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Trace the accuracy-vs-cost curve over a threshold grid.

    The sentinel 1.01 (> any probability) represents 'never stop early',
    i.e. the always-k_max policy the current manager has collapsed to.
    """
    if grid is None:
        grid = [round(i * 0.02, 2) for i in range(51)] + [1.01]
    probs_list = [example_stage_probs(rec, probe) for rec in records]
    return [simulate_threshold(records, probs_list, t) for t in grid]


def choose_threshold(
    sweep: List[Dict[str, Any]],
    acc_full: float,
    epsilon: float = 0.005,
) -> Dict[str, Any]:
    """Pick the operating point: min avg tool calls s.t. acc >= acc_full - eps.

    Falls back to the max-accuracy point (then min cost) when no point
    satisfies the constraint, and flags that in the result.
    """
    feasible = [pt for pt in sweep if pt["accuracy"] >= acc_full - epsilon]
    satisfied = bool(feasible)
    pool = feasible if feasible else list(sweep)
    if not pool:
        raise ValueError("Empty threshold sweep.")
    if feasible:
        best = sorted(
            pool,
            key=lambda pt: (pt["avg_tool_calls"], -pt["accuracy"], pt["threshold"]),
        )[0]
    else:
        best = sorted(
            pool,
            key=lambda pt: (-pt["accuracy"], pt["avg_tool_calls"], pt["threshold"]),
        )[0]
    return {
        "threshold": best["threshold"],
        "accuracy": best["accuracy"],
        "avg_tool_calls": best["avg_tool_calls"],
        "acc_full": round(float(acc_full), 4),
        "epsilon": epsilon,
        "constraint_satisfied": satisfied,
    }


# ---------------------------------------------------------------------------
# Reference policies (baselines / bounds) from the same counterfactual data
# ---------------------------------------------------------------------------

def fixed_k_stats(records: List[Dict[str, Any]], k_max: int = K_MAX_DEFAULT) -> Dict[str, Any]:
    """Accuracy of 'always stop after exactly k tools' for each k."""
    out: Dict[str, Any] = {}
    for k in range(k_max + 1):
        n = 0
        n_correct = 0
        for rec in records:
            stages = rec.get("stages") or []
            if not stages:
                continue
            idx = min(k, len(stages) - 1)
            n += 1
            if _stage_correct(rec, idx):
                n_correct += 1
        out[str(k)] = {"n": n, "accuracy": round(n_correct / max(1, n), 4)}
    return out


def oracle_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-question oracle: stop at the smallest k whose answer is correct.

    Upper-bounds any stopping policy operating on these trajectories.
    Questions never answered correctly are charged 0 tools (stop immediately).
    """
    n = 0
    n_correct = 0
    k_sum = 0
    for rec in records:
        stages = rec.get("stages") or []
        if not stages:
            continue
        n += 1
        correct_idxs = [i for i in range(len(stages)) if _stage_correct(rec, i)]
        if correct_idxs:
            n_correct += 1
            k_sum += correct_idxs[0]
    return {
        "n": n,
        "accuracy": round(n_correct / max(1, n), 4),
        "avg_tool_calls": round(k_sum / max(1, n), 4),
    }


def heuristic_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Two probe-free heuristics for comparison.

    stop_on_stable:    stop at the first k > 0 whose greedy answer equals the
                       previous stage's answer (else last stage).
    stop_on_unanimous: stop at the first stage whose votes are unanimous and
                       all parseable (else last stage).
    """
    def _simulate(pick_stop) -> Dict[str, Any]:
        n = 0
        n_correct = 0
        k_sum = 0
        for rec in records:
            stages = rec.get("stages") or []
            if not stages:
                continue
            idx = pick_stop(rec, stages)
            n += 1
            k_sum += idx
            if _stage_correct(rec, idx):
                n_correct += 1
        return {
            "n": n,
            "accuracy": round(n_correct / max(1, n), 4),
            "avg_tool_calls": round(k_sum / max(1, n), 4),
        }

    def _stable(rec, stages):
        prev = None
        for i, st in enumerate(stages):
            pred = st.get("pred")
            if i > 0 and pred is not None and prev is not None and str(pred) == str(prev):
                return i
            prev = pred
        return len(stages) - 1

    def _unanimous(rec, stages):
        n_choices = int(rec.get("n_choices") or 1)
        for i, st in enumerate(stages):
            vs = vote_stats(st.get("votes") or [], n_choices)
            if vs["agreement_top"] >= 1.0 and vs["valid_vote_frac"] >= 1.0:
                return i
        return len(stages) - 1

    return {
        "stop_on_stable": _simulate(_stable),
        "stop_on_unanimous": _simulate(_unanimous),
    }


def marginal_tool_table(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-step marginal benefit by tool kind.

    For step i (the tool executed between stage i and stage i+1):
      gain      = wrong -> correct
      loss      = correct -> wrong
      net_gain  = (gains - losses) / n_steps for that tool
    This is the direct empirical answer to 'what does each extra call buy'.
    """
    by_tool: Dict[str, Dict[str, int]] = {}
    for rec in records:
        stages = rec.get("stages") or []
        steps = rec.get("steps") or []
        for i, step in enumerate(steps):
            if i + 1 >= len(stages):
                continue
            kind = str(step.get("kind") or step.get("tool_name") or "unknown")
            slot = by_tool.setdefault(kind, {"n": 0, "gain": 0, "loss": 0})
            before = _stage_correct(rec, i)
            after = _stage_correct(rec, i + 1)
            slot["n"] += 1
            if after and not before:
                slot["gain"] += 1
            elif before and not after:
                slot["loss"] += 1
    out: Dict[str, Any] = {}
    for kind, s in sorted(by_tool.items()):
        out[kind] = {
            "n_steps": s["n"],
            "gain_rate": round(s["gain"] / max(1, s["n"]), 4),
            "loss_rate": round(s["loss"] / max(1, s["n"]), 4),
            "net_gain": round((s["gain"] - s["loss"]) / max(1, s["n"]), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Train/holdout splitting
# ---------------------------------------------------------------------------

def stable_holdout_split(
    records: List[Dict[str, Any]],
    holdout_frac: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic per-question split so probe fit and threshold selection
    never see the same questions. Keys on question_hash when present, else
    example_id."""
    train: List[Dict[str, Any]] = []
    hold: List[Dict[str, Any]] = []
    cut = max(0.0, min(1.0, holdout_frac)) * 100.0
    for rec in records:
        key = str(rec.get("question_hash") or rec.get("example_id") or "")
        bucket = zlib.crc32(key.encode("utf-8")) % 100
        (hold if bucket < cut else train).append(rec)
    return train, hold


# ---------------------------------------------------------------------------
# Trajectory text hygiene
# ---------------------------------------------------------------------------

_ANSWER_LINE_RE = re.compile(r"(?m)^(\s*)ANSWER_(?=[A-Za-z0-9_])")


def answer_to_draft(text: str) -> str:
    """Rewrite line-leading ANSWER_X into DRAFT_ANSWER_X.

    Used when an assistant turn that produced a final answer is retained in
    the history of a continuing trajectory: the policy forbids a final
    ANSWER_ line before the last turn, so keeping it verbatim would teach /
    condition on a malformed history. DRAFT_ANSWER_ lines are untouched
    (they don't start with 'ANSWER_').
    """
    if not text:
        return text
    return _ANSWER_LINE_RE.sub(r"\1DRAFT_ANSWER_", text)


# ---------------------------------------------------------------------------
# Distillation export (probe-selected minimal correct trajectories -> SFT)
# ---------------------------------------------------------------------------

def pick_stop_index(record: Dict[str, Any], probe: StopProbe, threshold: float) -> int:
    """First stage with P >= threshold, else the last recorded stage."""
    stages = record.get("stages") or []
    if not stages:
        return 0
    probs = example_stage_probs(record, probe)
    for i, p in enumerate(probs):
        if p >= threshold:
            return i
    return len(stages) - 1


def build_distill_rows(
    record: Dict[str, Any],
    k_star: int,
    require_correct: bool = True,
) -> List[Dict[str, Any]]:
    """Convert one counterfactual record truncated at stage k_star into
    per-turn SFT rows in the exact format of manager/evolve.py:

      {"example_id", "question_hash", "prompt": [messages...],
       "response": [one assistant message]}

    Intermediate turns teach the recorded tool call (with the model's own
    draft as content); the final turn teaches DRAFT+ANSWER built from the
    stage-k_star answer. Returns [] when the record cannot yield a clean
    trajectory (missing prediction, wrong answer under require_correct, or
    fewer recorded steps than k_star).
    """
    stages = record.get("stages") or []
    steps = record.get("steps") or []
    base_messages = record.get("base_messages") or []
    if not stages or not base_messages:
        return []
    k_star = max(0, min(int(k_star), len(stages) - 1))
    final_stage = stages[k_star]
    pred = final_stage.get("pred")
    if pred is None:
        return []
    if require_correct and str(pred) != str(record.get("ground_truth")):
        return []
    if len(steps) < k_star:
        return []

    eid = record.get("example_id")
    qhash = record.get("question_hash")
    rows: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = [dict(m) for m in base_messages]

    for i in range(k_star):
        step = steps[i]
        tool_name = str(step.get("tool_name") or "")
        call_id = str(step.get("call_id") or f"distill_{eid}_{i}")
        args = dict(step.get("args") or {})
        stage_pred_i = stages[i].get("pred")
        content = str(step.get("assistant_content") or "")
        if not content and stage_pred_i is not None:
            content = f"DRAFT_ANSWER_{_label_to_token(str(stage_pred_i))}"
        asst: Dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        }
        if content:
            asst["content"] = content
        rows.append({
            "example_id": eid,
            "question_hash": qhash,
            "prompt": [dict(m) for m in history],
            "response": [asst],
        })
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": str(step.get("tool_output") or ""),
        }
        history = history + [asst, tool_msg]

    token = _label_to_token(str(pred))
    final_text = f"ANSWER_{token}"
    if k_star == 0:
        response_content = final_text
    else:
        response_content = f"DRAFT_ANSWER_{token}\n{final_text}"
    rows.append({
        "example_id": eid,
        "question_hash": qhash,
        "prompt": [dict(m) for m in history],
        "response": [{"role": "assistant", "content": response_content}],
    })
    return rows
