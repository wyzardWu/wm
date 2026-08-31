#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

validate_common_environment
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: cache preparation needs at least one visible GPU" >&2
  exit 1
fi

CACHE_VARIANTS="${CACHE_VARIANTS:-v1,v2,v3}"
IFS=',' read -r -a REQUESTED_VARIANTS <<< "${CACHE_VARIANTS}"
if [[ "${#REQUESTED_VARIANTS[@]}" -eq 0 ]]; then
  echo "ERROR: CACHE_VARIANTS must select at least one variant" >&2
  exit 1
fi
for variant in "${REQUESTED_VARIANTS[@]}"; do
  case "${variant}" in
    v1|v2|v3) ;;
    *)
      echo "ERROR: unknown CACHE_VARIANTS entry: ${variant}" >&2
      exit 1
      ;;
  esac
done

variant_requested() {
  local candidate="$1"
  local variant
  for variant in "${REQUESTED_VARIANTS[@]}"; do
    if [[ "${variant}" == "${candidate}" ]]; then
      return 0
    fi
  done
  return 1
}

cd "${REPO_ROOT}"
"${PYTHON_BIN}" examples/Rebuttal/prepare_metadata.py
"${PYTHON_BIN}" examples/Rebuttal/cache_layout.py init --cache_base "${CACHE_BASE}"

prepare_variant() {
  local variant="$1"
  local metadata="$2"
  local cache_root="${CACHE_BASE}/${variant}"
  local -a worker_pids=()

  echo "Preparing ${variant}: ${metadata}"
  for rank in "${!REBUTTAL_GPUS[@]}"; do
    local gpu="${REBUTTAL_GPUS[$rank]}"
    local log="${cache_root}/cache-rank-${rank}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" examples/Rebuttal/precompute_cache_worker.py \
      --metadata "${metadata}" \
      --dataset_base "${DATA_ROOT}" \
      --cache_root "${cache_root}" \
      --model_paths "${CACHE_MODEL_PATHS}" \
      --tokenizer_path "${TOKENIZER_PATH}" \
      --rank "${rank}" \
      --world_size "${NUM_GPUS}" \
      --height 480 \
      --width 608 \
      --num_frames 101 \
      --skip_existing \
      >"${log}" 2>&1 &
    worker_pids+=("$!")
  done

  local failed=0
  for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "ERROR: one or more ${variant} cache workers failed; inspect ${cache_root}/cache-rank-*.log" >&2
    exit 1
  fi

  "${PYTHON_BIN}" examples/Rebuttal/cache_layout.py finalize \
    --cache_base "${CACHE_BASE}" \
    --variant "${variant}" \
    --metadata "${metadata}" \
    --dataset_base "${DATA_ROOT}" \
    --vae_path "${VAE_PATH}" \
    --t5_path "${T5_PATH}" \
    --height 480 \
    --width 608 \
    --num_frames 101 \
    >"${cache_root}/finalize_manifest.log"
  echo "Finalized ${cache_root}/manifest.json"
}

if variant_requested v1; then
  prepare_variant v1 "${REBUTTAL_DIR}/generated/metadata_v1_vanilla_strategy.csv"
fi
if variant_requested v2; then
  prepare_variant v2 "${REBUTTAL_DIR}/generated/metadata_v2_strategy_only.csv"
fi
if variant_requested v3; then
  prepare_variant v3 "${DATA_ROOT}/metadata.csv"
fi

echo "Selected rebuttal caches are complete: ${CACHE_VARIANTS}"
