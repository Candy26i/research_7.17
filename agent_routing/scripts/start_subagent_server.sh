#!/usr/bin/env bash
# Launch vLLM to serve the three frozen subagents on GPU 0.
#
# Prerequisites:
#   conda activate vllm_env   (separate env with: pip install vllm)
#
# Usage:
#   bash scripts/start_subagent_server.sh <base_model> <teacher_id> [output_root] [port] [max_model_len]
#
# Example:
#   bash scripts/start_subagent_server.sh Qwen/Qwen3.5-9B openai_us4_500_runtime_raw
#   bash scripts/start_subagent_server.sh Qwen/Qwen3.5-9B openai_us4_500_runtime_raw outputs 8000 32768

set -e

BASE_MODEL=${1:?"Usage: $0 <base_model> <teacher_id> [output_root] [port]"}
TEACHER_ID=${2:?"Usage: $0 <base_model> <teacher_id> [output_root] [port]"}
OUTPUT_ROOT=${3:-"outputs"}
PORT=${4:-8000}
MAX_MODEL_LEN=${5:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

ADAPTER_ROOT="${OUTPUT_ROOT}/adapters/${TEACHER_ID}"
EXTRACTOR="${ADAPTER_ROOT}/extractor_adapter"
REASONER="${ADAPTER_ROOT}/reasoner_adapter"
VERIFIER="${ADAPTER_ROOT}/verifier_adapter"

echo "[vLLM] base_model = ${BASE_MODEL}"
echo "[vLLM] teacher_id = ${TEACHER_ID}"
echo "[vLLM] extractor  = ${EXTRACTOR}"
echo "[vLLM] reasoner   = ${REASONER}"
echo "[vLLM] verifier   = ${VERIFIER}"
echo "[vLLM] port       = ${PORT}"
echo "[vLLM] max len    = ${MAX_MODEL_LEN}"
echo "[vLLM] memory use = ${GPU_MEMORY_UTILIZATION}"
echo "[vLLM] GPU        = 0"

# Verify adapter directories exist before launching.
for DIR in "${EXTRACTOR}" "${REASONER}" "${VERIFIER}"; do
    if [ ! -d "${DIR}" ]; then
        echo "[vLLM] ERROR: adapter directory not found: ${DIR}"
        exit 1
    fi
done

EXTRA_MODEL_ARGS=()
case "${BASE_MODEL,,}" in
  *qwen3.5*|*qwen3_5*)
    # This project is text-only. Skip the vision encoder/profiling and enable
    # Qwen3.5 reasoning parsing while keeping each sub-agent explicitly
    # non-thinking through chat_template_kwargs in RemoteSubagentPool.
    EXTRA_MODEL_ARGS+=(--language-model-only --reasoning-parser qwen3 --override-generation-config '{"enable_thinking": false}')
    ;;
esac

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --served-model-name base \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules \
        extractor="${EXTRACTOR}" \
        reasoner="${REASONER}" \
        verifier="${VERIFIER}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "${EXTRA_MODEL_ARGS[@]}"
