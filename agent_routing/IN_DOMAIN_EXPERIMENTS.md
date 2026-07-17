# In-Domain Experiment Manual — Adaptive Deliberation Control

This document is the canonical execution plan. Every benchmark is trained and
evaluated independently. **There is no cross-domain manager transfer in the
main experiment.** Each benchmark receives its own sub-agent adapters,
cold-start adapter, GRPO checkpoints, held-out questions, W&B namespace, and
statistical report.

## 1. Final design

| Benchmark | Training source | Sub-agent / cold-start / GRPO unique questions | Dev | Held-out pool | Final report |
|---|---|---:|---:|---:|---:|
| MedQA | Official train | 800 / 400 / 1,800 | 300 | 500 | 300 |
| LegalBench large5 | Remaining rows after balanced holdout | 600 / 300 / remainder (~1,234) | 200 (40/task) | 500 (100/task) | 300 (60/task) |
| MMLU-Pro | Frozen custom in-domain partition | 900 / 400 / 2,700 | 300 | 500 | 300 |
| GPQA | Extended minus every Diamond question | 100 / 40 / 178 | 30 | all Diamond 198 | all 198 |

All partitions use `seed=42`. Sub-agent SFT, cold-start, GRPO, dev, and test are
question-hash disjoint. The three sub-agent roles share the same sub-agent-SFT
question set within a benchmark; this allows paired specialization and subset
analysis without consuming three times as many unique questions.

MMLU-Pro is a custom in-domain partition of the public test corpus. It must not
be described as standard leaderboard evaluation.

## 2. One-time environment setup

```bash
cd /home/yizzhao/research_0703/agent_routing
source /home/yizzhao/research_0703/.venv/bin/activate
pip install -r requirements.txt accelerate deepspeed requests

export PYTHONUTF8=1
export BASE_MODEL=Qwen/Qwen3.5-9B   # use Qwen/Qwen3.5-4B for the smaller scale
export SEED=42
export DEEPSEEK_BASE_URL=http://localhost:8001/v1
export DEEPSEEK_MODEL=deepseek-ai/DeepSeek-V3

# Qwen3.5-4B tolerates the slightly larger LoRA SFT rate; keep 9B conservative.
if [[ "$BASE_MODEL" == *"3.5-4B"* ]]; then
  export SUBAGENT_SFT_LR=1e-4
else
  export SUBAGENT_SFT_LR=5e-5
fi

export SUBAGENT_EXTRACTOR_MAX_NEW_TOKENS=512
export SUBAGENT_REASONER_MAX_NEW_TOKENS=1024
export SUBAGENT_VERIFIER_MAX_NEW_TOKENS=768
```

The local DeepSeek endpoint must accept OpenAI-compatible
`/v1/chat/completions`. If a hosted endpoint is used, change
`DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL`, and `OPENAI_API_KEY`.

Run DeepSeek synthesis before starting the sub-agent server if both use GPU 0.
After the three sub-agent adapters are trained, stop DeepSeek and start:

```bash
conda activate vllm_env
bash scripts/start_subagent_server.sh "$BASE_MODEL" <TEACHER_ID> outputs 8000 32768
curl -f http://localhost:8000/health
```

The server exposes four model names:

```text
base        — base manager used for stepwise draft elicitation
extractor   — Extractor LoRA
reasoner    — Reasoner LoRA
verifier    — Verifier LoRA
```

## 3. Shared quality rules

Every sub-agent pipeline follows:

1. Generate GT-blind base-manager predictions on the sub-agent question pool.
2. Export more synthetic prompts than will be retained.
3. Use actual manager predictions as verifier candidates.
4. Import with schema, coverage, and symmetric choice-leakage checks.
5. Select a deterministic unique-question subset.
6. Audit before SFT.
7. Exclude selected sub-agent questions from cold-start and GRPO by question hash.

Cold-start uses:

```text
25% format-diverse trajectories
25% correctness-minus-cost oracle sequences
50% teacher sequences when configured, otherwise deterministic heuristic plans
```

The default draft mode is `base_stepwise`: after every sub-agent result,
including the last one, the base manager is queried again. This records every
W→C/C→W transition and provides the updated state before either another
delegation or stopping. `oracle` drafts are allowed only as an ablation.

## 3.1 Thinking mode and token-budget policy

The main ADC manager is intentionally **non-thinking**. This must be stated in
the paper; it studies whether external, specialized deliberation can help a
short-budget Manager route and stop. The main GRPO command has a 1,024-token
trajectory cap. Final learned-policy evaluation uses at most 256 Manager tokens
per turn and 1,024 Manager tokens total across at most four turns. Free-form
Manager reasoning is prohibited; it emits only `DRAFT_ANSWER_`, native tool
calls, and the terminal `ANSWER_` line.

The main paper uses framework-internal comparisons rather than an unrestricted
single-agent thinking baseline:

| Arm | Manager budget | Sub-agent budget | Purpose |
|---|---:|---:|---|
| learned ADC | 256/turn, 1,024 total | role-specific | main adaptive policy |
| forced none | 256 final turn | none | zero-delegation control |
| forced single/pair/all | 256 final turn | role-specific | fixed-subset and oracle analysis |

Role-specific completion caps are Extractor 512, Reasoner 1,024, and Verifier
768. Every report records actual prompt/completion/total tokens, cap hits,
complete MAS totals, cache hits, and accuracy/token usage conditioned on the
number of tools called. A smaller number of calls must reduce actual consumed
tokens; allocated context capacity is not counted as consumed compute.

Run the suite only after the GPU-0 sub-agent server is healthy:

```bash
bash scripts/run_eval_budget_suite.sh \
  "$ID" <EVAL_N> outputs/manager/$ID/grpo_anytime_c05 \
  "$BASE_MODEL" "$DESC" <BENCHMARK_DATA_FLAGS>
```

`--eval_enable_thinking` is explicit. If the tokenizer cannot honor it, the
evaluator fails instead of silently producing a mislabeled non-thinking run.
Use `--mgr_enable_thinking` only for a separately named manager-training
ablation; it is not part of the main ADC result.

Before any final test run, perform the Manager-cap check on development rows
only. Pass benchmark flags that make `test` empty so the evaluator selects
`dev`; never tune these caps on the final 300 rows or GPQA-Diamond:

```bash
for SPEC in "128 512 short" "256 1024 main" "512 2048 loose"; do
  read -r TURN TOTAL TAG <<< "$SPEC"
  CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" --teacher_id "$ID" <DEV_ONLY_DATA_FLAGS> \
    --eval_n_samples <DEV_N> \
    --eval_manager_dir outputs/manager/$ID/grpo_anytime_c05 \
    --subagent_server_url http://localhost:8000 \
    --eval_max_new_tokens "$TURN" \
    --eval_max_total_manager_tokens "$TOTAL" \
    --eval_out_tag "manager_budget_$TAG" --task_description "$DESC"
done
```

Keep the 256/1,024 main setting when both Manager cap-hit rate and total-budget
exhaustion are below 2%. Increase it only after inspecting actual truncated
outputs. For GPQA, use the non-Diamond dev rows and select by formatting/cap
diagnostics rather than noisy 30-question accuracy.

---

# 4. MedQA — complete pipeline

## 4.1 Variables and split

```bash
export ID=adc_medqa_ds
export DATA=outputs/data/medqa_us4_normalized.jsonl
export DESC="You are a manager agent solving USMLE-style medical multiple-choice questions."
export DARGS="--medqa_normalized_cache $DATA --train_size 3000 --dev_size 300 --test_size 500 --seed $SEED"

python -m src.pipeline.cli load_medqa \
  --base_model "$BASE_MODEL" --medqa_normalized_cache "$DATA" \
  --medqa_refresh_cache --train_size 3000 --dev_size 300 --test_size 500 --seed "$SEED"
```

Expected split: `3000 / 300 / 500`.

## 4.2 Base predictions for verifier candidates

Generate predictions for the same first 1,600 shuffled training questions used
by prompt export:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli export_base_predictions \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
  --base_prediction_n_samples 1600 \
  --base_prediction_out outputs/sft_data/$ID/base_predictions.jsonl \
  --task_description "$DESC"
```

## 4.3 Export, generate, import, and select synthetic sub-agent data

```bash
for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli export_deepseek_jsonl \
    --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
    --agent_kind "$KIND" --n_samples 1600 \
    --synth_verifier_candidate_jsonl outputs/sft_data/$ID/base_predictions.jsonl \
    --deepseek_prompt_jsonl outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl

  python scripts/generate_openai_compatible_jsonl.py \
    --input outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl \
    --output outputs/sft_data/$ID/${KIND}_responses_raw.jsonl \
    --base_url "$DEEPSEEK_BASE_URL" --model "$DEEPSEEK_MODEL" \
    --workers 8 --temperature 0.4 --max_tokens 2200

  python -m src.pipeline.cli import_deepseek_jsonl \
    --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind "$KIND" \
    --deepseek_prompt_jsonl outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl \
    --deepseek_response_jsonl outputs/sft_data/$ID/${KIND}_responses_raw.jsonl \
    --deepseek_sft_jsonl outputs/sft_data/$ID/${KIND}_sft_filtered_raw.jsonl \
    --synth_symmetric_leakage

done

python scripts/select_shared_synthetic_rows.py \
  --extractor outputs/sft_data/$ID/extractor_sft_filtered_raw.jsonl \
  --reasoner outputs/sft_data/$ID/reasoner_sft_filtered_raw.jsonl \
  --verifier outputs/sft_data/$ID/verifier_sft_filtered_raw.jsonl \
  --output_dir outputs/sft_data/$ID --n 800 --seed "$SEED"

python scripts/audit_synthetic_data.py \
  outputs/sft_data/$ID/extractor_sft_final.jsonl \
  outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --min_rows 800 --require_verifier_candidates
```

## 4.4 Train and gate the sub-agents

```bash
for KIND in extractor reasoner verifier; do
  CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli train_subagent \
    --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind "$KIND" \
    --sft_train_jsonl outputs/sft_data/$ID/${KIND}_sft_final.jsonl \
    --sft_epochs 3 --sft_lr "$SUBAGENT_SFT_LR" --sft_bs 1 --sft_grad_accum 8
done

CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli eval_subagents \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS --eval_n_samples 100
```

Require JSON/schema validity at least 0.90 before manager training.

## 4.5 Stepwise mixed cold-start

Terminal A:

```bash
conda activate vllm_env
bash scripts/start_subagent_server.sh "$BASE_MODEL" "$ID" outputs 8000 32768
```

Terminal B:

```bash
source /home/yizzhao/research_0703/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli manager_coldstart_sft \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
  --exclude_sft_example_ids outputs/sft_data/$ID/extractor_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --coldstart_n_samples 400 \
  --coldstart_draft_source base_stepwise \
  --coldstart_draft_server_url http://localhost:8000 \
  --subagent_server_url http://localhost:8000 \
  --coldstart_sequence_policy mixed \
  --coldstart_oracle_cost_per_tool 0.05 \
  --manager_sft_epochs 2 --manager_sft_lr 5e-6 \
  --task_description "$DESC"
```

Inspect `outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl` and report
W→C, C→C, W→W, and C→W counts. If drafts never change, stop before GRPO and
debug the stepwise manager endpoint.

```bash
python scripts/audit_coldstart_trajectories.py \
  outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl
```

## 4.6 GRPO

```bash
bash scripts/train_manager_grpo_multigpu.sh "$ID" \
  --base_model "$BASE_MODEL" $DARGS \
  --exclude_sft_example_ids outputs/sft_data/$ID/extractor_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --exclude_sft_example_ids outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl \
  --mgr_init_adapter outputs/manager/$ID/sft_coldstart \
  --mgr_output_dir outputs/manager/$ID/grpo_anytime_c05 \
  --mgr_adc_mode --mgr_adc_variant anytime \
  --mgr_adc_draft_bonus 0.2 --mgr_adc_missing_draft_penalty 0.1 \
  --mgr_adc_final_bonus 1.0 --mgr_adc_cost_per_tool 0.05 \
  --mgr_clip_epsilon_high 0.28 --mgr_max_steps 300 \
  --mgr_use_wandb --wandb_project adc_in_domain \
  --wandb_run_name ${ID}_grpo_c05 --task_description "$DESC"
```

## 4.7 Final 300-question evaluation

```bash
bash scripts/run_eval_budget_suite.sh \
  "$ID" 300 outputs/manager/$ID/grpo_anytime_c05 \
  "$BASE_MODEL" "$DESC" $DARGS
```

Run all forced subsets on the same 300 questions:

```bash
for SEQ in none extractor reasoner verifier \
  extractor,reasoner extractor,verifier reasoner,verifier \
  extractor,reasoner,verifier; do
  CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli eval_manager_forced \
    --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
    --eval_n_samples 300 --eval_forced_tools "$SEQ" \
    --eval_max_new_tokens 256 \
    --eval_manager_dir outputs/manager/$ID/grpo_anytime_c05 \
    --subagent_server_url http://localhost:8000 \
    --eval_out_tag "${SEQ//,/_}" --task_description "$DESC"
done
```

---

# 5. LegalBench large5 — complete pipeline

## 5.1 Build the 2,834-row balanced partition

```bash
export ID=adc_legal_large5_ds
export RAW=outputs/data/legalbench_large5_raw.jsonl
export DATA=outputs/data/legalbench_large5_split.jsonl
export CONFIGS=corporate_lobbying,definition_classification,consumer_contracts_qa,canada_tax_court_outcomes,function_of_decision_section
export DESC="You are a manager agent solving LegalBench legal classification tasks."

python -m src.pipeline.cli load_legalbench \
  --base_model "$BASE_MODEL" --legalbench_configs "$CONFIGS" \
  --legalbench_normalized_cache "$RAW" --legalbench_refresh_cache \
  --train_size 0 --dev_size 0 --test_size 0 --seed "$SEED"

python scripts/build_legalbench_large5_splits.py \
  --input "$RAW" --output "$DATA" --test_per_task 100 --dev_per_task 40 --seed "$SEED"

export DARGS="--legalbench_configs $CONFIGS --legalbench_normalized_cache $DATA --train_size 2200 --dev_size 200 --test_size 500 --seed $SEED"
```

The split script prints the exact train count. `train_size=2200` is only a cap;
if 2,134 rows remain, all 2,134 are used.

## 5.2 Base predictions and balanced synthetic data

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli export_base_predictions \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
  --base_prediction_n_samples 1200 \
  --base_prediction_stratify_by task_subtype \
  --base_prediction_out outputs/sft_data/$ID/base_predictions.jsonl \
  --task_description "$DESC"

for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli export_deepseek_jsonl \
    --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
    --agent_kind "$KIND" --n_samples 1200 --synth_stratify_by task_subtype \
    --synth_verifier_candidate_jsonl outputs/sft_data/$ID/base_predictions.jsonl \
    --deepseek_prompt_jsonl outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl

  python scripts/generate_openai_compatible_jsonl.py \
    --input outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl \
    --output outputs/sft_data/$ID/${KIND}_responses_raw.jsonl \
    --base_url "$DEEPSEEK_BASE_URL" --model "$DEEPSEEK_MODEL" \
    --workers 8 --temperature 0.4 --max_tokens 2200

  python -m src.pipeline.cli import_deepseek_jsonl \
    --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind "$KIND" \
    --deepseek_prompt_jsonl outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl \
    --deepseek_response_jsonl outputs/sft_data/$ID/${KIND}_responses_raw.jsonl \
    --deepseek_sft_jsonl outputs/sft_data/$ID/${KIND}_sft_filtered_raw.jsonl \
    --synth_symmetric_leakage

done

python scripts/select_shared_synthetic_rows.py \
  --extractor outputs/sft_data/$ID/extractor_sft_filtered_raw.jsonl \
  --reasoner outputs/sft_data/$ID/reasoner_sft_filtered_raw.jsonl \
  --verifier outputs/sft_data/$ID/verifier_sft_filtered_raw.jsonl \
  --output_dir outputs/sft_data/$ID --n 600 \
  --balance_by task_subtype --seed "$SEED"

python scripts/audit_synthetic_data.py \
  outputs/sft_data/$ID/extractor_sft_final.jsonl \
  outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --min_rows 600 --require_verifier_candidates
```

The selected files must contain 120 rows per task per sub-agent.

## 5.3 Sub-agent SFT, cold-start, and GRPO

```bash
for KIND in extractor reasoner verifier; do
  CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli train_subagent \
    --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind "$KIND" \
    --sft_train_jsonl outputs/sft_data/$ID/${KIND}_sft_final.jsonl \
    --sft_epochs 3 --sft_lr "$SUBAGENT_SFT_LR" --sft_bs 1 --sft_grad_accum 8
done

CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli eval_subagents \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS --eval_n_samples 100

# Terminal A (after sub-agent evaluation):
conda activate vllm_env
bash scripts/start_subagent_server.sh "$BASE_MODEL" "$ID" outputs 8000 32768
curl -f http://localhost:8000/health

# Terminal B:
source /home/yizzhao/research_0703/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli manager_coldstart_sft \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
  --exclude_sft_example_ids outputs/sft_data/$ID/extractor_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --coldstart_n_samples 300 --coldstart_draft_source base_stepwise \
  --coldstart_draft_server_url http://localhost:8000 \
  --subagent_server_url http://localhost:8000 \
  --coldstart_sequence_policy mixed --coldstart_oracle_cost_per_tool 0.05 \
  --manager_sft_epochs 2 --manager_sft_lr 5e-6 --task_description "$DESC"

python scripts/audit_coldstart_trajectories.py \
  outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl

bash scripts/train_manager_grpo_multigpu.sh "$ID" \
  --base_model "$BASE_MODEL" $DARGS \
  --exclude_sft_example_ids outputs/sft_data/$ID/extractor_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --exclude_sft_example_ids outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl \
  --mgr_init_adapter outputs/manager/$ID/sft_coldstart \
  --mgr_output_dir outputs/manager/$ID/grpo_anytime_c05 \
  --mgr_adc_mode --mgr_adc_variant anytime --mgr_adc_cost_per_tool 0.05 \
  --mgr_adc_draft_bonus 0.2 --mgr_adc_missing_draft_penalty 0.1 \
  --mgr_adc_final_bonus 1.0 --mgr_clip_epsilon_high 0.28 --mgr_max_steps 210 \
  --mgr_use_wandb --wandb_project adc_in_domain \
  --wandb_run_name ${ID}_grpo_c05 --task_description "$DESC"
```

## 5.4 Balanced final evaluation: 60 per task

```bash
bash scripts/run_eval_budget_suite.sh \
  "$ID" 300 outputs/manager/$ID/grpo_anytime_c05 \
  "$BASE_MODEL" "$DESC" $DARGS --eval_per_task 60
```

Use `--eval_per_task 60 --eval_n_samples 300` for every LegalBench baseline and
forced-subset run. Report `by_task`, macro task accuracy, overall accuracy, and
per-task average tool calls.

```bash
for SEQ in none extractor reasoner verifier \
  extractor,reasoner extractor,verifier reasoner,verifier \
  extractor,reasoner,verifier; do
  CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli eval_manager_forced \
    --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
    --eval_per_task 60 --eval_n_samples 300 --eval_forced_tools "$SEQ" \
    --eval_max_new_tokens 256 \
    --eval_manager_dir outputs/manager/$ID/grpo_anytime_c05 \
    --subagent_server_url http://localhost:8000 \
    --eval_out_tag "${SEQ//,/_}" --task_description "$DESC"
done
```

---

# 6. MMLU-Pro — complete custom in-domain pipeline

## 6.1 Freeze the custom split

```bash
export ID=adc_mmlupro_ds
export RAW=outputs/data/mmlu_pro_normalized.jsonl
export DATA=outputs/data/mmlu_pro_custom_split.jsonl
export DESC="You are a manager agent solving multiple-choice questions across diverse academic subjects. Each question has up to 10 options (A-J)."

python -m src.pipeline.cli load_mmlu_pro \
  --base_model "$BASE_MODEL" --mmlu_pro_normalized_cache "$RAW" \
  --mmlu_pro_splits test --mmlu_pro_refresh_cache \
  --train_size 0 --dev_size 0 --test_size 12032 --seed "$SEED"

python scripts/build_mmlu_pro_splits.py \
  --input "$RAW" --output "$DATA" \
  --train_size 4000 --dev_size 300 --test_size 500 --seed "$SEED"

export DARGS="--mmlu_pro_normalized_cache $DATA --train_size 4000 --dev_size 300 --test_size 500 --mmlu_pro_allow_test_training --seed $SEED"
```

## 6.2 Base predictions and synthetic data

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli export_base_predictions \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
  --base_prediction_n_samples 1800 \
  --base_prediction_stratify_by metadata:category \
  --base_prediction_out outputs/sft_data/$ID/base_predictions.jsonl \
  --task_description "$DESC"

for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli export_deepseek_jsonl \
    --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
    --agent_kind "$KIND" --n_samples 1800 --synth_stratify_by metadata:category \
    --synth_verifier_candidate_jsonl outputs/sft_data/$ID/base_predictions.jsonl \
    --deepseek_prompt_jsonl outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl

  python scripts/generate_openai_compatible_jsonl.py \
    --input outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl \
    --output outputs/sft_data/$ID/${KIND}_responses_raw.jsonl \
    --base_url "$DEEPSEEK_BASE_URL" --model "$DEEPSEEK_MODEL" \
    --workers 8 --temperature 0.4 --max_tokens 2200

  python -m src.pipeline.cli import_deepseek_jsonl \
    --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind "$KIND" \
    --deepseek_prompt_jsonl outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl \
    --deepseek_response_jsonl outputs/sft_data/$ID/${KIND}_responses_raw.jsonl \
    --deepseek_sft_jsonl outputs/sft_data/$ID/${KIND}_sft_filtered_raw.jsonl \
    --synth_symmetric_leakage

done

python scripts/select_shared_synthetic_rows.py \
  --extractor outputs/sft_data/$ID/extractor_sft_filtered_raw.jsonl \
  --reasoner outputs/sft_data/$ID/reasoner_sft_filtered_raw.jsonl \
  --verifier outputs/sft_data/$ID/verifier_sft_filtered_raw.jsonl \
  --output_dir outputs/sft_data/$ID --n 900 \
  --balance_by stratum --seed "$SEED"

python scripts/audit_synthetic_data.py \
  outputs/sft_data/$ID/extractor_sft_final.jsonl \
  outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --min_rows 900 --require_verifier_candidates
```

## 6.3 Sub-agent SFT, cold-start, GRPO, and final evaluation

```bash
for KIND in extractor reasoner verifier; do
  CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli train_subagent \
    --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind "$KIND" \
    --sft_train_jsonl outputs/sft_data/$ID/${KIND}_sft_final.jsonl \
    --sft_epochs 3 --sft_lr "$SUBAGENT_SFT_LR" --sft_bs 1 --sft_grad_accum 8
done

CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli eval_subagents \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS --eval_n_samples 100

# Terminal A:
conda activate vllm_env
bash scripts/start_subagent_server.sh "$BASE_MODEL" "$ID" outputs 8000 32768
curl -f http://localhost:8000/health

# Terminal B:
source /home/yizzhao/research_0703/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli manager_coldstart_sft \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
  --exclude_sft_example_ids outputs/sft_data/$ID/extractor_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --coldstart_n_samples 400 --coldstart_draft_source base_stepwise \
  --coldstart_draft_server_url http://localhost:8000 \
  --subagent_server_url http://localhost:8000 \
  --coldstart_sequence_policy mixed --coldstart_oracle_cost_per_tool 0.05 \
  --manager_sft_epochs 2 --manager_sft_lr 5e-6 --task_description "$DESC"

python scripts/audit_coldstart_trajectories.py \
  outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl

bash scripts/train_manager_grpo_multigpu.sh "$ID" \
  --base_model "$BASE_MODEL" $DARGS \
  --exclude_sft_example_ids outputs/sft_data/$ID/extractor_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --exclude_sft_example_ids outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl \
  --mgr_init_adapter outputs/manager/$ID/sft_coldstart \
  --mgr_output_dir outputs/manager/$ID/grpo_anytime_c05 \
  --mgr_adc_mode --mgr_adc_variant anytime --mgr_adc_cost_per_tool 0.05 \
  --mgr_adc_draft_bonus 0.2 --mgr_adc_missing_draft_penalty 0.1 \
  --mgr_adc_final_bonus 1.0 --mgr_clip_epsilon_high 0.28 --mgr_max_steps 450 \
  --mgr_use_wandb --wandb_project adc_in_domain \
  --wandb_run_name ${ID}_grpo_c05 --task_description "$DESC"

bash scripts/run_eval_budget_suite.sh \
  "$ID" 300 outputs/manager/$ID/grpo_anytime_c05 \
  "$BASE_MODEL" "$DESC" $DARGS

for SEQ in none extractor reasoner verifier \
  extractor,reasoner extractor,verifier reasoner,verifier \
  extractor,reasoner,verifier; do
  CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli eval_manager_forced \
    --base_model "$BASE_MODEL" --teacher_id "$ID" $DARGS \
    --eval_n_samples 300 --eval_forced_tools "$SEQ" \
    --eval_max_new_tokens 256 \
    --eval_manager_dir outputs/manager/$ID/grpo_anytime_c05 \
    --subagent_server_url http://localhost:8000 \
    --eval_out_tag "${SEQ//,/_}" --task_description "$DESC"
done
```

The paper must label this result `MMLU-Pro custom in-domain split (n=300)`.

---

# 7. GPQA — train on all non-Diamond, evaluate all Diamond

## 7.1 Build the only allowed split

```bash
export ID=adc_gpqa_nondiamond_ds
export TRAIN=outputs/data/gpqa_nondiamond_train348.jsonl
export TEST=outputs/data/gpqa_diamond_eval198.jsonl
export DESC="You are a manager agent solving expert-level graduate science multiple-choice questions."

python scripts/build_gpqa_splits.py --eval_n 198 --seed "$SEED" --out_dir outputs/data

export TRAIN_ARGS="--gpqa_normalized_cache $TRAIN --train_size 318 --dev_size 30 --test_size 0 --seed $SEED"
export TEST_ARGS="--gpqa_normalized_cache $TEST --train_size 0 --dev_size 0 --test_size 198 --seed $SEED"
```

No Diamond question may appear in sub-agent synthesis, base predictions,
cold-start, GRPO, dev selection, prompt tuning, or reward tuning.

## 7.2 Base predictions and synthetic data

The verifier-specific rationale, artifact contract, preflight checks, and
recovery steps are documented in
`outputs/sft_data/adc_gpqa_nondiamond_ds/README.md`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli export_base_predictions \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $TRAIN_ARGS \
  --base_prediction_n_samples 200 \
  --base_prediction_out outputs/sft_data/$ID/base_predictions.jsonl \
  --task_description "$DESC"

for KIND in extractor reasoner verifier; do
  python -m src.pipeline.cli export_deepseek_jsonl \
    --base_model "$BASE_MODEL" --teacher_id "$ID" $TRAIN_ARGS \
    --agent_kind "$KIND" --n_samples 200 \
    --synth_verifier_candidate_jsonl outputs/sft_data/$ID/base_predictions.jsonl \
    --deepseek_prompt_jsonl outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl

done

python scripts/audit_deepseek_prompts.py \
  outputs/sft_data/$ID/extractor_prompts_raw.jsonl \
  outputs/sft_data/$ID/reasoner_prompts_raw.jsonl \
  outputs/sft_data/$ID/verifier_prompts_raw.jsonl \
  --min_rows 190 --require_verifier_candidates \
  --min_verifier_candidate_coverage 1.0

for KIND in extractor reasoner verifier; do
  python scripts/generate_openai_compatible_jsonl.py \
    --input outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl \
    --output outputs/sft_data/$ID/${KIND}_responses_raw.jsonl \
    --base_url "$DEEPSEEK_BASE_URL" --model "$DEEPSEEK_MODEL" \
    --workers 4 --temperature 0.4 --max_tokens 3000

  python -m src.pipeline.cli import_deepseek_jsonl \
    --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind "$KIND" \
    --deepseek_prompt_jsonl outputs/sft_data/$ID/${KIND}_prompts_raw.jsonl \
    --deepseek_response_jsonl outputs/sft_data/$ID/${KIND}_responses_raw.jsonl \
    --deepseek_sft_jsonl outputs/sft_data/$ID/${KIND}_sft_filtered_raw.jsonl \
    --synth_symmetric_leakage

done

python scripts/select_shared_synthetic_rows.py \
  --extractor outputs/sft_data/$ID/extractor_sft_filtered_raw.jsonl \
  --reasoner outputs/sft_data/$ID/reasoner_sft_filtered_raw.jsonl \
  --verifier outputs/sft_data/$ID/verifier_sft_filtered_raw.jsonl \
  --output_dir outputs/sft_data/$ID --n 100 --seed "$SEED"

python scripts/audit_synthetic_data.py \
  outputs/sft_data/$ID/extractor_sft_final.jsonl \
  outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --min_rows 100 --require_verifier_candidates
```

## 7.3 Sub-agent SFT, 40-question cold-start, and GRPO

```bash
for KIND in extractor reasoner verifier; do
  CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli train_subagent \
    --base_model "$BASE_MODEL" --teacher_id "$ID" --agent_kind "$KIND" \
    --sft_train_jsonl outputs/sft_data/$ID/${KIND}_sft_final.jsonl \
    --sft_epochs 4 --sft_lr "$SUBAGENT_SFT_LR" --sft_bs 1 --sft_grad_accum 8
done

CUDA_VISIBLE_DEVICES=0 python -m src.pipeline.cli eval_subagents \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $TRAIN_ARGS --eval_n_samples 30

# Terminal A:
conda activate vllm_env
bash scripts/start_subagent_server.sh "$BASE_MODEL" "$ID" outputs 8000 32768
curl -f http://localhost:8000/health

# Terminal B:
source /home/yizzhao/research_0703/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli manager_coldstart_sft \
  --base_model "$BASE_MODEL" --teacher_id "$ID" $TRAIN_ARGS \
  --exclude_sft_example_ids outputs/sft_data/$ID/extractor_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --coldstart_n_samples 40 --coldstart_draft_source base_stepwise \
  --coldstart_draft_server_url http://localhost:8000 \
  --subagent_server_url http://localhost:8000 \
  --coldstart_sequence_policy mixed --coldstart_oracle_cost_per_tool 0.05 \
  --manager_sft_epochs 3 --manager_sft_lr 5e-6 --task_description "$DESC"

python scripts/audit_coldstart_trajectories.py \
  outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl

bash scripts/train_manager_grpo_multigpu.sh "$ID" \
  --base_model "$BASE_MODEL" $TRAIN_ARGS \
  --exclude_sft_example_ids outputs/sft_data/$ID/extractor_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/reasoner_sft_final.jsonl \
  --exclude_sft_example_ids outputs/sft_data/$ID/verifier_sft_final.jsonl \
  --exclude_sft_example_ids outputs/manager/$ID/evolve/manager_sft_coldstart.jsonl \
  --mgr_init_adapter outputs/manager/$ID/sft_coldstart \
  --mgr_output_dir outputs/manager/$ID/grpo_anytime_c05 \
  --mgr_adc_mode --mgr_adc_variant anytime --mgr_adc_cost_per_tool 0.05 \
  --mgr_adc_draft_bonus 0.2 --mgr_adc_missing_draft_penalty 0.1 \
  --mgr_adc_final_bonus 1.0 --mgr_grpo_beta 0.001 \
  --mgr_clip_epsilon_high 0.28 --mgr_max_steps 60 \
  --mgr_use_wandb --wandb_project adc_in_domain \
  --wandb_run_name ${ID}_grpo_c05 --task_description "$DESC"
```

## 7.4 Evaluate all 198 Diamond questions

```bash
bash scripts/run_eval_budget_suite.sh \
  "$ID" 198 outputs/manager/$ID/grpo_anytime_c05 \
  "$BASE_MODEL" "$DESC" $TEST_ARGS

for SEQ in none extractor reasoner verifier \
  extractor,reasoner extractor,verifier reasoner,verifier \
  extractor,reasoner,verifier; do
  CUDA_VISIBLE_DEVICES=1 python -m src.pipeline.cli eval_manager_forced \
    --base_model "$BASE_MODEL" --teacher_id "$ID" $TEST_ARGS \
    --eval_n_samples 198 --eval_forced_tools "$SEQ" \
    --eval_max_new_tokens 256 \
    --eval_manager_dir outputs/manager/$ID/grpo_anytime_c05 \
    --subagent_server_url http://localhost:8000 \
    --eval_out_tag "${SEQ//,/_}" --task_description "$DESC"
done
```

Use all 198 questions for every baseline and forced-subset run. Three stochastic
runs are still `198 unique questions × 3 runs`, not 594 independent examples.

---

# 8. Mandatory baselines and ablations

Run these within each benchmark on exactly the same held-out question IDs:

1. Base Qwen3.5-9B direct.
2. Base + self-consistency with matched token budget.
3. Direct CoT distillation control using the same DeepSeek token budget.
4. Forced `none`, each single sub-agent, each two-sub-agent subset, and all three.
5. Learned ADC policy.
6. Correctness-cost per-question oracle over the eight forced subsets.

For cold-start ablation, compare:

```text
base_stepwise + mixed sequences       main
base_initial + mixed sequences        no state refresh
base_stepwise + diverse sequences     no oracle action labels
oracle drafts + diverse sequences     old privileged baseline
```

For reward ablation, compare:

```text
anytime       main
binary        final correctness only
transition    expected sandbagging failure
sum           expected unnecessary-call farming
```

For cost Pareto curves, retrain each main manager with:

```text
cost_per_tool ∈ {0.02, 0.05, 0.10, 0.20}
```

# 9. Reporting checklist

For every benchmark and seed report:

- final accuracy and exact 95% bootstrap confidence interval;
- average tool calls and tool-call rate;
- extractor/reasoner/verifier call frequencies;
- initial-draft accuracy;
- W→C correction rate, C→W corruption rate, W→W and C→C rates;
- fixed-subset accuracy/cost points;
- learned-policy reward and per-question oracle regret;
- AURC or risk-coverage curve;
- latency and generated tokens;
- question hashes used by train/dev/test.

Use three training seeds: `42, 43, 44`. Accuracy comparisons on identical
questions use McNemar's test. Tool counts, reward, AURC, and regret use paired
question bootstrap. LegalBench additionally reports all five task results and
their macro average.

Do not select a final checkpoint, prompt, cost, or stopping threshold using the
held-out pool. All choices are made on dev; held-out evaluation is run only
after the configuration is frozen.
