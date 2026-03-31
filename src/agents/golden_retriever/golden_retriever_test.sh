#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_TRACE_PATH="${PROJECT_ROOT}/trace_logs/planner/planner_test_loong_set4_en_travelplanner.json"
PLANNER_TRACE_PATH="${1:-${DEFAULT_TRACE_PATH}}"

if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "${PLANNER_TRACE_PATH}" ]]; then
  echo "Planner trace not found: ${PLANNER_TRACE_PATH}" >&2
  exit 1
fi

python "${SCRIPT_DIR}/golden_retriever_test.py" \
  --planner_trace_path "${PLANNER_TRACE_PATH}" \
  "$@"
