#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

validate_common_environment
SMOKE_GPU="${SMOKE_GPU:-${REBUTTAL_GPUS[0]}}"
SMOKE_OUTPUT_BASE="${SMOKE_OUTPUT_BASE:-${OUTPUT_BASE}/smoke}"

cd "${REPO_ROOT}"
for variant in v1 v2 v3; do
  output="${SMOKE_OUTPUT_BASE}/${variant}"
  echo "Running two-step smoke test: ${variant} -> ${output}"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" PYTHONUNBUFFERED=1 \
    "${ACCELERATE_BIN}" launch \
    --num_processes=1 \
    --main_process_port="$((29800 + ${variant#v}))" \
    examples/Rebuttal/train.py \
    --variant "${variant}" \
    --dataset_base_path "${DATA_ROOT}" \
    --output_path "${output}" \
    --wan_root "${WAN_ROOT}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --smoke_test \
    --expected_num_processes 1
done
