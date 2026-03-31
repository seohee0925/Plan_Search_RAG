#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-e2e_set1_sweep}"
MAX_PER_COMBO="${2:-1}"

python /workspace/golden_retriever/src/end-to-end_test.py \
  --set_id 1 \
  --max_per_combo "${MAX_PER_COMBO}" \
  --max_items 100 \
  --sample_prefix "${PREFIX}"
