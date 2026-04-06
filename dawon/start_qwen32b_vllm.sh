#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/structrag/bin/python}
MODEL_PATH=${MODEL_PATH:-/workspace/StructRAG/model/Qwen2.5-32B-Instruct}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-1225}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen}
LOG_PATH=${LOG_PATH:-$SCRIPT_DIR/logs/qwen32b_vllm.log}

mkdir -p "$(dirname "$LOG_PATH")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --model "$MODEL_PATH" \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 1 \
  --guided-decoding-backend lm-format-enforcer \
  --enforce-eager \
  --disable-custom-all-reduce \
  2>&1 | tee "$LOG_PATH"
