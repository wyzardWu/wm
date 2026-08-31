#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

cache_variant="${CACHE_VARIANT:-}"
case "${cache_variant}" in
  v2)
    metadata="${REBUTTAL_DIR}/generated/metadata_v2_strategy_only.csv"
    ;;
  v3)
    metadata="${DATA_ROOT}/metadata.csv"
    ;;
  *)
    echo "ERROR: CACHE_VARIANT must be v2 or v3; got ${cache_variant:-<empty>}" >&2
    exit 1
    ;;
esac
variant_label="${cache_variant^^}"

validate_common_environment
if [[ "${CUDA_VISIBLE_DEVICES}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: ${variant_label} T5 cache expects CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ "${NUM_GPUS}" -ne 8 ]]; then
  echo "ERROR: ${variant_label} T5 cache expects eight workers; got ${NUM_GPUS}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
cache_root="${CACHE_BASE}/${cache_variant}"
require_file "${metadata}"

"${PYTHON_BIN}" examples/Rebuttal/cache_layout.py init \
  --cache_base "${CACHE_BASE}" >/dev/null
mkdir -p "${cache_root}/_workers"

run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
declare -a worker_pids=()

echo "Starting ${variant_label} T5-only cache with eight workers"
echo "metadata=${metadata}"
echo "cache=${cache_root}"
echo "GPUs=${CUDA_VISIBLE_DEVICES}"

for rank in "${!REBUTTAL_GPUS[@]}"; do
  gpu="${REBUTTAL_GPUS[$rank]}"
  log="${cache_root}/t5-rank-${rank}-${run_tag}.log"
  echo "Launching rank=${rank} on physical GPU ${gpu}; log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    nice -n 10 \
    "${PYTHON_BIN}" examples/Rebuttal/precompute_cache_worker.py \
    --metadata "${metadata}" \
    --dataset_base "${DATA_ROOT}" \
    --cache_root "${cache_root}" \
    --model_paths "[\"${T5_PATH}\"]" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --rank "${rank}" \
    --world_size "${NUM_GPUS}" \
    --height 480 \
    --width 608 \
    --num_frames 101 \
    --skip_existing \
    --t5_only \
    >"${log}" 2>&1 &
  worker_pids+=("$!")
done

echo "Worker PIDs: ${worker_pids[*]}"
failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "ERROR: one or more ${variant_label} T5 workers failed; inspect ${cache_root}/t5-rank-*-${run_tag}.log" >&2
  exit 1
fi

"${PYTHON_BIN}" examples/Rebuttal/cache_layout.py finalize \
  --cache_base "${CACHE_BASE}" \
  --variant "${cache_variant}" \
  --metadata "${metadata}" \
  --dataset_base "${DATA_ROOT}" \
  --vae_path "${VAE_PATH}" \
  --t5_path "${T5_PATH}" \
  --height 480 \
  --width 608 \
  --num_frames 101 \
  >"${cache_root}/finalize_manifest.log"

echo "${variant_label} T5 cache complete and manifest finalized: ${cache_root}/manifest.json"
