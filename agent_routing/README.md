# Agent Routing — Learning When to Commit

A pipeline for training a manager LLM (Qwen3-8B) that learns **when to stop
delegating and commit to an answer**. At every step the manager faces four
actions:

```
delegate(extractor) | delegate(reasoner) | delegate(verifier) | COMMIT
```

The three advisors are frozen, schema-constrained specialists that provide
*signals, never answers*; the manager is the sole authority on the final
`ANSWER_<TOKEN>` line and states a `DRAFT_ANSWER_<TOKEN>` (its current belief)
in every delegating turn. The stopping policy is trained with GRPO under an
**anytime-accuracy reward** designed to be incentive-compatible: honestly
reporting your current best guess is the unique optimal drafting strategy.

Benchmarks: **MedQA-USMLE**, **LegalBench**, **MMLU-Pro**, **GPQA**.

> Paper-experiment matrix (RQ1/RQ2/RQ3 tables, ablation arms, oracle/regret
> analysis, data-budget rationale): **[EXPERIMENTS.md](EXPERIMENTS.md)**.
> This README covers the system + one end-to-end walkthrough per benchmark.

---

## Architecture

```
                       ┌─────────────────────────────┐
                       │   Manager (Qwen3-8B)        │
                       │   GRPO + anytime ADC reward │
                       │   drafts -> delegate/commit │
                       └────┬────────┬───────┬───────┘
                            │        │       │ current_draft
              ┌─────────────┘        │       └──────────────┐
              ▼                      ▼                      ▼
   ┌──────────────────┐   ┌─────────────────┐   ┌──────────────────────┐
   │  ExtractorAgent  │   │  ReasonerAgent  │   │  VerifierAgent       │
   │  (frozen, LoRA)  │   │ (frozen, LoRA)  │   │  (frozen, LoRA;      │
   │  key signals     │   │ neutral scaffold│   │  audits the draft)   │
   └────────┬─────────┘   └────────┬────────┘   └──────────┬───────────┘
            │                      │                       │
            └──────────────────────┼───────────────────────┘
                                   │
                   ┌───────────────▼───────────────┐
                   │  Teacher (GPT / Claude /      │
                   │  DeepSeek) — schema-gated     │
                   │  synthesis of advisor SFT data│
                   └───────────────────────────────┘
```

**Hard invariants**
- Advisors never produce the final answer (pydantic schemas + leakage audit at
  synthesis; `--synth_symmetric_leakage` audits all choice texts).
- Advisors are frozen, greedy-decoded, and cached per (kind, question) — the
  marginal value of every consultation is deterministic, which makes the
  per-question stopping oracle enumerable (`eval_manager_forced`).
- Each advisor is callable at most once per episode; each call costs
  `--mgr_adc_cost_per_tool`.

**The reward**

```
R = final_bonus       * 1[final answer correct]
  + draft_bonus       * (mean correctness of ALL answer statements)   # bounded
  - missing_draft_pen * max(0, #delegations - #drafts)                # format only
  - cost_per_tool     * #delegations
```

Two natural alternatives are provably exploitable and exist only as ablation
arms (`--mgr_adc_variant transition|sum`): transition rewards telescope into
paying for a deliberately wrong first draft (sandbagging); summed draft
bonuses are farmable by superfluous delegations. See `src/manager/reward.py`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt accelerate deepspeed
# separate env for the advisor server (8B multi-GPU runs)
conda create -n vllm_env python=3.11 -y && conda activate vllm_env && pip install vllm
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / DEEPSEEK_API_KEY
export PYTHONUTF8=1
```

Multi-GPU layout for 8B (single GPU works for 0.6B/4B without the server):

```
GPU 0  →  vLLM: base + 3 LoRA advisors     bash scripts/start_subagent_server.sh <base> <teacher_id>
GPU 1-3 → DeepSpeed manager training       bash scripts/train_manager_grpo_multigpu.sh <teacher_id> [flags]
```

---

# End-to-end walkthroughs, one per benchmark

Common shape of every pipeline: **load → synthesize advisor data ×3 → SFT
advisors ×3 → gate → manager cold start → GRPO → eval**. Only the data flags
and budgets change. `--teacher_id` namespaces all outputs, so runs never
collide.

## 1. MedQA (primary training domain — full pipeline)

```bash
export BASE_MODEL=Qwen/Qwen3-8B
export TEACHER_ID=commit_gpt_8b
export PROVIDER=openai MODEL=gpt-4o
export MEDQA_CACHE=outputs/data/medqa_us4_normalized.jsonl
export TASK_DESC="You are a manager agent solving USMLE-style medical multiple-choice questions."
export SPLIT="--train_size 1400 --dev_size 200 --test_size 500"

# 1) data
python -m src.pipeline.cli load_medqa --base_model "$BASE_MODEL" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT

# 2) synthesize advisor SFT data (500 each; verifier also samples audit candidates)
for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli synth_subagent \
      --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
      --teacher_provider "$PROVIDER" --teacher_model "$MODEL" \
      --agent_kind "$KIND" --n_samples 500 --synth_symmetric_leakage \
      --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT \
      --task_description "$TASK_DESC"
done

# 3) SFT the three advisors
for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" --agent_kind "$KIND" \
      --sft_epochs 3 --sft_lr 5e-5 --sft_bs 1 --sft_grad_accum 8
done

# 4) gate (json_ok_rate & schema_ok_rate > 0.9)
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT --eval_n_samples 50

# 5) manager cold start (demonstrates DRAFT_ANSWER_ + current_draft)
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/verifier_sft.jsonl" \
    --coldstart_n_samples 300 --coldstart_force_diverse \
    --manager_sft_epochs 2 --manager_sft_lr 5e-6 \
    --task_description "$TASK_DESC"

# 6) GRPO — terminal A: advisor server on GPU 0
conda activate vllm_env
bash scripts/start_subagent_server.sh "$BASE_MODEL" "$TEACHER_ID"

# 6) GRPO — terminal B: training on GPUs 1-3 (anytime reward, cost 0.05)
EXCL="--exclude_sft_example_ids outputs/sft_data/${TEACHER_ID}/extractor_sft.jsonl \
      --exclude_sft_example_ids outputs/sft_data/${TEACHER_ID}/reasoner_sft.jsonl \
      --exclude_sft_example_ids outputs/sft_data/${TEACHER_ID}/verifier_sft.jsonl \
      --exclude_sft_example_ids outputs/manager/${TEACHER_ID}/evolve/manager_sft_coldstart_diverse.jsonl"
bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
    --base_model "$BASE_MODEL" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT $EXCL \
    --mgr_init_adapter "outputs/manager/${TEACHER_ID}/sft_coldstart" \
    --mgr_output_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
    --mgr_adc_mode --mgr_adc_variant anytime \
    --mgr_adc_draft_bonus 0.2 --mgr_adc_missing_draft_penalty 0.1 \
    --mgr_adc_final_bonus 1.0 --mgr_adc_cost_per_tool 0.05 \
    --mgr_clip_epsilon_high 0.28 --mgr_max_steps 300 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TEACHER_ID}_anytime_c05" \
    --task_description "$TASK_DESC"

# 7) eval (learned delegate-or-commit loop, 500 held-out)
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache "$MEDQA_CACHE" $SPLIT --eval_n_samples 500 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
    --task_description "$TASK_DESC"
```

## 2. LegalBench

**Recommended use: zero-shot probe of a MedQA-trained manager** (the 5-task
pool is only ~380 rows — an honest in-domain pipeline barely fits):

```bash
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --legalbench_configs "abercrombie,hearsay,personal_jurisdiction,proa,successor_liability" \
    --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl \
    --train_size 0 --dev_size 0 --test_size 400 --eval_n_samples 400 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
    --task_description "You are a manager agent solving LegalBench legal classification tasks."
# report per task: filter the eval jsonl by task_subtype
```

**Optional in-domain training** — the same seven steps as MedQA with scaled
budgets. Add more configs to enlarge the pool if you can; with 5 tasks use:

```bash
export LB_ID=commit_gpt_lb_8b
export LB="--legalbench_configs abercrombie,hearsay,personal_jurisdiction,proa,successor_liability \
           --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl"
export LB_SPLIT="--train_size 240 --dev_size 50 --test_size 95"
export LB_DESC="You are a manager agent solving LegalBench legal classification tasks."
# step 2: --n_samples 120        (per advisor)
# step 5: --coldstart_n_samples 60
# step 6: --mgr_max_steps 60 --mgr_grpo_beta 0.02   (tiny GRPO pool ~50 rows; watch for overfit)
# steps otherwise identical to MedQA with $LB $LB_SPLIT --teacher_id $LB_ID --task_description "$LB_DESC"
```

## 3. MMLU-Pro

**Recommended use: zero-shot probe** (the community compares on the full test
split; training on any part of it breaks comparability):

```bash
python -m src.pipeline.cli load_mmlu_pro --base_model "$BASE_MODEL" \
    --mmlu_pro_normalized_cache outputs/data/mmlu_pro_normalized.jsonl \
    --mmlu_pro_splits test --train_size 0 --dev_size 0 --test_size 500

python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --mmlu_pro_normalized_cache outputs/data/mmlu_pro_normalized.jsonl \
    --train_size 0 --dev_size 0 --test_size 500 --eval_n_samples 500 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
    --task_description "You are a manager agent solving multiple-choice questions across diverse academic subjects. Each question has 10 options (A-J)."
```

**Optional in-domain training** (accepting the comparability caveat): identical
seven steps to MedQA with `--mmlu_pro_normalized_cache ... --train_size 1800
--dev_size 200 --test_size 500`, `--n_samples 500`, `--coldstart_n_samples 300`,
`--mgr_max_steps 300`, under a fresh `--teacher_id`.

## 4. GPQA

One-time: accept the dataset terms on HuggingFace, then `huggingface-cli login`.

**Zero-shot probe on Diamond (all 198 questions):**

```bash
python -m src.pipeline.cli load_gpqa --base_model "$BASE_MODEL" \
    --gpqa_subsets gpqa_diamond \
    --gpqa_normalized_cache outputs/data/gpqa_diamond_normalized.jsonl \
    --train_size 0 --dev_size 0 --test_size 198

python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$TEACHER_ID" \
    --gpqa_normalized_cache outputs/data/gpqa_diamond_normalized.jsonl \
    --train_size 0 --dev_size 0 --test_size 198 --eval_n_samples 198 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_anytime_c05" \
    --task_description "You are a manager agent solving expert-level graduate science multiple-choice questions."
```

**In-domain training** — GPQA is 546 questions total (nested subsets), so the
split is built once by a script: eval = 100 held-out diamond questions, train
pool = 446 (98 leftover diamond + 348 non-diamond, hash-disjoint):

```bash
python scripts/build_gpqa_splits.py --eval_n 100 --seed 42
# -> outputs/data/gpqa_diamond_eval100.jsonl  (eval, never trained on)
# -> outputs/data/gpqa_train446.jsonl         (advisor SFT 160 + cold start 50 + dev 40 + GRPO ~196)
```

Then run the same seven steps as MedQA with
`--gpqa_normalized_cache outputs/data/gpqa_train446.jsonl --train_size 0
--dev_size 40 --test_size 0`, budgets 160/50, `--mgr_max_steps 120
--mgr_grpo_beta 0.02`, and evaluate ONLY on `gpqa_diamond_eval100.jsonl`.
Full copy-paste commands: EXPERIMENTS.md §8.4.

---

## Pipeline stages (reference)

| Stage | What it does |
|---|---|
| `load_medqa` / `load_gpqa` / `load_mmlu_pro` | download + normalize (GPQA: `--gpqa_exclude_subsets` for nested-subset dedup) |
| `synth_subagent` | teacher synthesis, four quality gates (JSON → schema → coverage → leakage) |
| `export_deepseek_jsonl` / `import_deepseek_jsonl` | offline-teacher alternative to synth |
| `train_subagent` | LoRA-SFT one advisor |
| `eval_subagents` | JSON/schema validity gate |
| `manager_coldstart_sft` | build + train tool-calling format (drafts + `current_draft` demonstrated) |
| `train_manager_grpo` | GRPO; `--mgr_adc_mode --mgr_adc_variant anytime\|transition\|sum` |
| `evolve_build_sft` / `train_manager_sft` / `evolve_round` | failure-recycling SFT loop |
| `eval_manager` | no-tools probe; `--eval_sc_k K` = self-consistency baseline |
| `eval_manager_tools` | full delegate-or-commit loop (the learned policy) |
| `eval_manager_forced` | fixed delegation subsets → fixed-k baselines + stopping oracle |

Troubleshooting, calibration/oracle analysis snippets, reward-ablation arms,
and the full paper-experiment matrix: **[EXPERIMENTS.md](EXPERIMENTS.md)**.
