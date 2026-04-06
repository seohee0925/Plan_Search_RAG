#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/structrag/bin/python}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-32B-Instruct}
MODEL_PATH=${MODEL_PATH:-/workspace/StructRAG/model/Qwen2.5-32B-Instruct}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  "$@"
