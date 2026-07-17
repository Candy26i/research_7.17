#!/usr/bin/env bash
# Run the learned routing policy on one benchmark's frozen evaluation IDs.
# Framework-internal forced subsets (including `none`) are run separately by
# IN_DOMAIN_EXPERIMENTS.md. Unrestricted single-agent thinking is deliberately
# outside the main scientific comparison.
#
# Usage:
#   bash scripts/run_eval_budget_suite.sh \
#     <run_id> <eval_n> <manager_dir> <base_model> <task_description> \
#     [benchmark/data flags...]

set -euo pipefail

RUN_ID=${1:?"run_id required"}
EVAL_N=${2:?"eval_n required"}
MANAGER_DIR=${3:?"manager_dir required"}
BASE_MODEL=${4:?"base_model required"}
TASK_DESC=${5:?"task_description required"}
shift 5

EVAL_GPU=${EVAL_GPU:-1}
SUBAGENT_SERVER_URL=${SUBAGENT_SERVER_URL:-http://localhost:8000}

COMMON=(
  --base_model "${BASE_MODEL}"
  --teacher_id "${RUN_ID}"
  --eval_n_samples "${EVAL_N}"
  --task_description "${TASK_DESC}"
  "$@"
)

# Learned ADC policy: 256 tokens per decision and 1,024 total across at most
# four Manager turns. Manager and sub-agent prompt/completion tokens are
# reported separately and together.
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python -m src.pipeline.cli eval_manager_tools \
  "${COMMON[@]}" --eval_manager_dir "${MANAGER_DIR}" \
  --eval_max_new_tokens 256 --eval_max_total_manager_tokens 1024 \
  --subagent_server_url "${SUBAGENT_SERVER_URL}" \
  --eval_out_tag adc_policy_1k
