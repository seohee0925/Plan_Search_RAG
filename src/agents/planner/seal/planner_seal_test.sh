#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PARQUET_PATH="${PROJECT_ROOT}/sealqa/longseal.parquet"

if [[ ! -f "${PARQUET_PATH}" ]]; then
  echo "LongSEAL parquet not found: ${PARQUET_PATH}" >&2
  exit 1
fi

python "${SCRIPT_DIR}/planner_seal_test.py" \
  --parquet_path "${PARQUET_PATH}" \
  --index 0 \
  --doc_field 30_docs \
  --sample_id "planner_seal_longseal_idx0_30_docs" \
  "$@"
