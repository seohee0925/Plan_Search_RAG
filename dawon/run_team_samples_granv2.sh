#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/workspace/venvs/structrag/bin/python}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-32B-Instruct}
MODEL_PATH=${MODEL_PATH:-/workspace/StructRAG/model/Qwen2.5-32B-Instruct}
SAMPLE_PREFIX=${SAMPLE_PREFIX:-dawon_team_samples_granv2}
MANIFEST_PATH=${MANIFEST_PATH:-$SCRIPT_DIR/team_samples_manifest.json}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_pipeline.py" \
  --manifest_path "$MANIFEST_PATH" \
  --sample_prefix "$SAMPLE_PREFIX" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --divider_mode granv2 \
  "$@"
