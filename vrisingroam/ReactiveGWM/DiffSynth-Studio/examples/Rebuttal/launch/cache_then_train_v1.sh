#!/usr/bin/env bash

set -euo pipefail
REBUTTAL_LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REBUTTAL_LAUNCH_DIR}/_common.sh"

if [[ "${NUM_GPUS}" -ne "${FORMAL_NUM_GPUS}" ]]; then
  echo "ERROR: V1 cache-to-train launch requires exactly ${FORMAL_NUM_GPUS} GPUs; got ${NUM_GPUS}: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

export PYTHON_BIN ACCELERATE_BIN DATA_ROOT WAN_ROOT TOKENIZER_PATH
export CACHE_BASE OUTPUT_BASE CUDA_VISIBLE_DEVICES
export CACHE_VARIANTS=v1

bash "${REBUTTAL_LAUNCH_DIR}/prepare_cache.sh"

V1_MANIFEST="${CACHE_BASE}/v1/manifest.json"
require_file "${V1_MANIFEST}"
echo "Validated V1 cache manifest: ${V1_MANIFEST}"

bash "${REBUTTAL_LAUNCH_DIR}/train_v1_vanilla_strategy.sh" "$@"
