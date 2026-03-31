#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PROCESSED_JSONL_PATH="${PROJECT_ROOT}/Loong/data/loong_process.jsonl"
RAW_JSONL_PATH="${PROJECT_ROOT}/Loong/data/loong.jsonl"

if [[ -f "${PROCESSED_JSONL_PATH}" ]]; then
  JSONL_PATH="${PROCESSED_JSONL_PATH}"
elif [[ -f "${RAW_JSONL_PATH}" ]]; then
  JSONL_PATH="${RAW_JSONL_PATH}"
else
  echo "Loong dataset not found: ${PROCESSED_JSONL_PATH} or ${RAW_JSONL_PATH}" >&2
  exit 1
fi

DOMAINS=("paper" "legal" "financial")

for domain in "${DOMAINS[@]}"; do
  python "${SCRIPT_DIR}/planner_test.py" \
    --jsonl_path "${JSONL_PATH}" \
    --type "${domain}" \
    --match_rank 0 \
    --require_question \
    --sample_id "planner_test_loong_domain_${domain}" \
    "$@"
done
