# Harness-Level Optimal Stopping ("Stop Probe")

**Goal.** Keep the binary-reward GRPO manager exactly as it is (it works), but
cut the average number of subagent calls at (near-)equal accuracy — i.e. stop
each episode at the point of maximal marginal utility. **Neither the RL
algorithm nor the reward function is modified by this change.**

**Why not reward shaping (first principles).** The stopping decision cannot be
learned reliably through the reward because:

1. **Counterfactual blindness** — a rollout that calls 3 tools and is correct
   never observes that 1 tool would have sufficed. Marginal benefit
   `Δ(k) = acc(k+1) − acc(k)` is defined per question, but a single trajectory
   only ever visits one `k`.
2. **No gradient after collapse** — once the policy collapses to always-3,
   there is zero variance of `k` inside a GRPO group, so the group-relative
   advantage carries no signal about stopping (this is why CCR and ADC
   tool-cost terms plateaued).
3. **Scalar rewards cannot encode lexicographic preferences** — "accuracy
   first, then fewer calls" has no robust scalarization: a small cost term is
   drowned by advantage noise, a large one trades away accuracy.

**The fix.** Move stopping out of the policy and into the harness:

- Measure `Δ(k)` directly with **forced-continuation counterfactual rollouts**
  (answer probed at every stage k = 0..3 of the *same* question).
- Fit a tiny **calibrated logistic stop probe**
  `P(current answer is correct | label-free stage features)` on those records
  (main features: self-consistency vote agreement, answer stability across
  stages, k, entropy, question stats).
- Deploy the rule **"stop at the first stage where P ≥ θ"**; sweep θ to trace
  the accuracy-vs-cost Pareto curve and pick the operating point
  *min tool calls s.t. accuracy ≥ full-budget accuracy − ε*.
- Optionally **distill** the probe-selected minimal correct trajectories back
  into the manager via the existing SFT path, so the deployed model stops
  early on its own with no probe at inference time.

---

## 1. What was changed

### New file: `src/manager/stopping.py`

Pure logic, no torch/transformers imports (unit-testable on CPU):

| Symbol | Purpose |
|---|---|
| `STOP_PROBE_USER_MSG` | The forced-answer nudge appended on the probe branch only. |
| `FEATURE_NAMES`, `stage_features(...)` | The 10 label-free per-stage features (k, vote agreement/entropy, greedy-vs-majority, stability vs previous stage, chance rate, lengths). Fitting and deployment share this single function, so feature drift is impossible. |
| `vote_stats(...)` | Deterministic vote aggregation (tie-break by count desc, key asc). |
| `StopProbe` | Standardized logistic model with save/load; refuses to load under a feature-schema mismatch; carries the recommended `threshold`. |
| `fit_stop_probe(...)`, `fit_logistic(...)` | Deterministic full-batch L2 logistic regression (numpy), plus AUC / Brier / ECE metrics. |
| `threshold_sweep`, `simulate_threshold`, `choose_threshold` | Offline policy simulation on counterfactual records; picks θ\* = min cost subject to `acc ≥ acc_full − ε` (falls back to max-accuracy and flags `constraint_satisfied: false` if infeasible). |
| `fixed_k_stats`, `oracle_stats`, `heuristic_stats` | Always-k baselines, the per-question stopping oracle (upper bound), and two probe-free heuristics (stop-on-stable, stop-on-unanimous). |
| `marginal_tool_table` | Per-tool wrong→correct / correct→wrong rates for each executed step — the direct empirical answer to "what does each extra call buy". |
| `stable_holdout_split` | Deterministic per-question split (crc32 of question hash) so θ selection never sees fitting questions. |
| `answer_to_draft` | Rewrites line-leading `ANSWER_X` → `DRAFT_ANSWER_X` when a would-be final turn is retained inside a continuing history (keeps histories policy-conformant). |
| `pick_stop_index`, `build_distill_rows` | Truncate a counterfactual record at the probe's stop stage and emit per-turn SFT rows in the **exact** `{"example_id","question_hash","prompt","response"}` format of `manager/evolve.py`, including tool-call turns with JSON-string arguments. Rejection-samples to correct-only by default. |

### Modified: `src/pipeline/stages.py`

Nothing existing was touched except adding `parse_draft_answer` and the
`stopping` import; four new stages are appended:

- **`run_collect_counterfactuals`** — drives the trained manager through a
  forced-continuation rollout per question: at every stage k = 0..k_max a
  *probe branch* (copy of the history + `STOP_PROBE_USER_MSG`) forces a greedy
  answer plus `n_votes` sampled answers; the *main trajectory* then lets the
  manager pick the next tool (valid unused native call) or injects the next
  unused tool in canonical order (extractor → reasoner → verifier) when the
  manager stops early or calls invalidly. Records everything needed
  downstream: `base_messages`, per-stage `{pred, votes, final_text, …}`,
  per-step `{tool, chosen_by, model_wanted_stop, args, tool_output, …}`.
  Output: `outputs/eval/<teacher_id>/counterfactual_<tag>.jsonl` (+ report
  with fixed-k accuracies, oracle, heuristics, marginal-tool table).
  Reuses the same manager loading, binding-mode resolution, tool schemas,
  chat rendering, and subagent pool (local or vLLM remote) as
  `run_eval_manager_tools` / `run_eval_manager_forced`.
- **`run_fit_stop_probe`** — fits the probe on counterfactual JSONL(s),
  evaluates on a held-out question split (or a separate `--probe_eval_jsonl`),
  sweeps thresholds, chooses θ\*, and writes
  `outputs/eval/<teacher_id>/stop_probe.json` + a full report (sweep curve,
  fixed-k, oracle, heuristics, calibration metrics).
- **`run_eval_manager_adaptive`** — deployment evaluation: the manager runs
  its normal free-choice loop, but before every potential tool call the
  harness runs the stage probe; at the first stage with `P ≥ θ` the episode is
  terminated and the probe's greedy answer is submitted. The manager's own
  early stops are honored by default; `--adaptive_force_continue` additionally
  overrides them while the probe is unconfident. Reports accuracy, average
  tool calls, stop reasons, per-stage probabilities, and probe generation
  overhead.
- **`run_export_stop_distill_sft`** — converts probe-selected minimal correct
  trajectories into `train_manager_sft`-compatible rows
  (`outputs/manager/<teacher_id>/evolve/stop_distill_sft.jsonl`).

### Modified: `src/pipeline/cli.py`

Four new stage names (`collect_counterfactuals`, `fit_stop_probe`,
`eval_manager_adaptive`, `export_stop_distill_sft`) and their flags
(`--cf_*`, `--probe_*`, `--stop_threshold`, `--adaptive_force_continue`,
`--distill_*`). All existing stages and flags are untouched.

### New file: `tests/test_stopping.py`

14 CPU-only tests covering: vote statistics (ties, invalid votes, entropy),
feature ordering and prev-pred chaining, probe fitting on synthetic data
(AUC > 0.9, monotonicity in agreement), save/load round-trip + schema guard,
threshold-sweep extremes (θ=0 ⇒ always k=0; sentinel ⇒ always k_max),
constrained threshold choice + infeasible fallback, fixed-k/oracle/heuristic
baselines, the marginal-tool table, deterministic holdout splitting,
`answer_to_draft` hygiene, and distillation-row structure/rejection rules.

**Verification performed:** `python -m pytest tests/test_stopping.py -q` →
14 passed; full import chain of `cli`/`stages`/`stopping` verified; and an
end-to-end offline smoke run of `run_fit_stop_probe` +
`run_export_stop_distill_sft` on 200 synthetic records confirmed the whole
chain (probe AUC 1.0 on separable data, θ\* satisfying the accuracy
constraint at 0.87 avg tools vs 3.0, distillation emitting k\*=0 for easy and
k\*=2 for hard questions with schema-valid SFT rows).

---

## 2. Experiment playbook

Notation: `$COMMON` = your usual context flags, e.g.
`--base_model <model> --teacher_id <id> --output_root outputs
--mmlu_pro_normalized_cache outputs/data/mmlu_pro_normalized.jsonl
--train_size 600 --dev_size 100 --test_size 200`
(swap the benchmark flags for GPQA/MedQA/LegalBench runs; keep the exact same
split flags across all steps so train/test membership is identical).
`--eval_manager_dir` should point at your best GRPO manager checkpoint;
add `--subagent_server_url http://…` if you serve subagents via vLLM.

### Step 0 — Smoke run (≈20 min GPU)

```bash
python -m src.pipeline.cli collect_counterfactuals $COMMON \
    --eval_manager_dir outputs/manager/<id>/grpo \
    --cf_split train --cf_n_samples 20 --cf_n_votes 3
```

Check `outputs/eval/<id>/counterfactual_train_report.json`:
`fixed_k_accuracy` should be populated for k = 0..3, `tool_choice` should show
mostly `model_chosen_steps` (your manager likes calling tools), and stage-3
accuracy should be close to your known `eval_manager_tools` accuracy. If
stage-k accuracies look degenerate (e.g. all-invalid answers at k=0), inspect
a few `final_text` fields in the JSONL — the probe nudge should elicit an
`ANSWER_` line.

### Step 1 — Collect counterfactuals (the marginal-utility dataset)

```bash
# Probe-fitting data: TRAIN split (never test).
python -m src.pipeline.cli collect_counterfactuals $COMMON \
    --eval_manager_dir outputs/manager/<id>/grpo \
    --cf_split train --cf_n_samples 600 --cf_n_votes 5

# Offline-analysis data: TEST split (used ONLY for reporting curves).
python -m src.pipeline.cli collect_counterfactuals $COMMON \
    --eval_manager_dir outputs/manager/<id>/grpo \
    --cf_split test --cf_n_samples 200 --cf_n_votes 5
```

Cost: per question ≈ 4 probe stages × (1 greedy + 5 votes) short generations
(≤512 tokens) + ≤3 continuation turns + ≤3 subagent calls. Subagent outputs
are cached per example, so reruns are cheap.

**First deliverable (before any probe exists):** the report's
`fixed_k_accuracy` and `marginal_tool_table` on the train file. This alone
answers "what does the 2nd and 3rd call actually buy, per tool" — expect
`net_gain` to shrink sharply with k if your marginal-returns hypothesis holds.

### Step 2 — Fit the probe and pick the operating point

```bash
python -m src.pipeline.cli fit_stop_probe $COMMON \
    --probe_train_jsonl outputs/eval/<id>/counterfactual_train.jsonl \
    --probe_epsilon 0.005
```

Read `outputs/eval/<id>/stop_probe_report.json` and gate on:

- `holdout_metrics.auc` ≥ ~0.75 (below that, raise `--cf_n_votes` to 8 or add
  more training questions; the agreement feature is the workhorse).
- `holdout_metrics.ece` ≤ ~0.08 (calibration is what makes θ meaningful).
- `threshold_choice.constraint_satisfied` = true and
  `threshold_choice.avg_tool_calls` clearly < 3.
- Compare `threshold_choice` against `heuristics.*` and `oracle` in the same
  report: probe should beat both heuristics and sit between them and the
  oracle on the cost axis.

The sweep array in the report is your **Pareto curve** (accuracy vs avg tool
calls, one point per θ) — this is the main figure.

### Step 3 — Offline generalization check on the test counterfactuals

```bash
python -m src.pipeline.cli fit_stop_probe $COMMON \
    --probe_train_jsonl outputs/eval/<id>/counterfactual_train.jsonl \
    --probe_eval_jsonl  outputs/eval/<id>/counterfactual_test.jsonl \
    --probe_out_tag test_sweep
```

**Protocol note:** ignore this run's *recommended* threshold (choosing θ on
test would leak); instead, look up the sweep row at the θ\* chosen in Step 2
and confirm accuracy/cost transfer from train-holdout to test.

### Step 4 — Online adaptive evaluation on test (headline numbers)

```bash
# Baseline (already have): free policy, expect ~3 tools/question.
python -m src.pipeline.cli eval_manager_tools $COMMON \
    --eval_manager_dir outputs/manager/<id>/grpo --eval_n_samples 200

# Adaptive harness stopping at the Step-2 threshold.
python -m src.pipeline.cli eval_manager_adaptive $COMMON \
    --eval_manager_dir outputs/manager/<id>/grpo \
    --probe_path outputs/eval/<id>/stop_probe.json \
    --eval_n_samples 200
```

Main table columns: accuracy, avg_tool_calls, valid_answer_rate, plus
`avg_probe_generations` for honest overhead accounting (votes are short
manager generations; subagent calls are the expensive resource you are
saving). Success criterion: accuracy within ε of `eval_manager_tools` at a
substantially lower avg_tool_calls; `stop_reasons` should be dominated by
`probe_confident`.

Recommended ablations (each is one flag):

- `--stop_threshold 0.3/0.5/0.7/0.9` → the *online* Pareto curve (should track
  the offline sweep — that agreement is itself a strong result).
- `--cf_n_votes 0/3/5/8` at fixed θ → confidence-signal ablation
  (`n_votes 0` isolates the votes' contribution; re-fit the probe per setting).
- `--adaptive_force_continue` → harness owning both stop directions.
- Existing baselines for the same table: `eval_manager_forced` with
  `none` / `reasoner` / `extractor,reasoner` / `extractor,reasoner,verifier`
  (fixed-k), and `eval_manager --eval_sc_k 6` (matched-compute
  self-consistency control).

### Step 5 (optional) — Distill the stopping rule into the manager

```bash
python -m src.pipeline.cli export_stop_distill_sft $COMMON \
    --distill_cf_jsonl outputs/eval/<id>/counterfactual_train.jsonl \
    --probe_path outputs/eval/<id>/stop_probe.json

python -m src.pipeline.cli train_manager_sft $COMMON \
    --manager_sft_train_jsonl outputs/manager/<id>/evolve/stop_distill_sft.jsonl \
    --manager_sft_epochs 1

# Re-evaluate WITHOUT the probe: has the model internalized early stopping?
python -m src.pipeline.cli eval_manager_tools $COMMON \
    --eval_manager_dir outputs/manager/<id>/sft_evolved --eval_n_samples 200
```

Success criterion: avg_tool_calls drops toward the distilled `k_star_dist`
average while accuracy holds — the harness improvement is now internalized
(no inference-time probe cost at all). Sanity-check `k_star_dist` in the
export report first: if it is ~all-zeros or ~all-threes, revisit θ.

### Pitfalls / invariants

- **Leakage rules:** probe weights fit on train questions only; θ chosen on
  train-holdout only; test counterfactuals are for reporting, never selection.
  Distillation input must be the *train* counterfactual file.
- Keep `--seed` and all split-size flags identical across steps 1–5 so the
  train/test partition is stable (the pipeline shuffles with `ctx.seed`).
- `collect_counterfactuals` overwrites its output file per tag — use
  `--cf_out_tag` to keep multiple versions side by side.
- The probe JSON is bound to `FEATURE_NAMES`; if you change the feature set,
  re-fit — stale probes refuse to load by design.
- ε (`--probe_epsilon`) is your accuracy budget; report results at 2–3 values
  (e.g. 0.0 / 0.005 / 0.01) rather than tuning it silently.
