#!/usr/bin/env bash
set -euo pipefail

exec vllm serve "$MODEL_PATH" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN"
