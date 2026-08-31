#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

validate_common_environment
SMOKE_GPU="${SMOKE_GPU:-${REBUTTAL_GPUS[0]}}"
SMOKE_SOURCE_BASE="${SMOKE_SOURCE_BASE:?Set SMOKE_SOURCE_BASE to a completed smoke_all output}"
RELOAD_OUTPUT_BASE="${RELOAD_OUTPUT_BASE:-${SMOKE_SOURCE_BASE}/reload}"

cd "${REPO_ROOT}"
for variant in v1 v2; do
  source_checkpoint="${SMOKE_SOURCE_BASE}/${variant}/step-1.safetensors"
  require_file "${source_checkpoint}"
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" PYTHONUNBUFFERED=1 \
    "${ACCELERATE_BIN}" launch \
    --num_processes=1 \
    --main_process_port="$((29900 + ${variant#v}))" \
    examples/Rebuttal/train.py \
    --variant "${variant}" \
    --dataset_base_path "${DATA_ROOT}" \
    --output_path "${RELOAD_OUTPUT_BASE}/${variant}" \
    --wan_root "${WAN_ROOT}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --smoke_test \
    --resume_checkpoint "${source_checkpoint}"
done

v3_manifest="${SMOKE_SOURCE_BASE}/v3/step-1.manifest.json"
require_file "${v3_manifest}"
CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" PYTHONUNBUFFERED=1 \
  "${ACCELERATE_BIN}" launch \
  --num_processes=1 \
  --main_process_port=29903 \
  examples/Rebuttal/train.py \
  --variant v3 \
  --dataset_base_path "${DATA_ROOT}" \
  --output_path "${RELOAD_OUTPUT_BASE}/v3" \
  --wan_root "${WAN_ROOT}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --smoke_test \
  --resume_manifest "${v3_manifest}"
