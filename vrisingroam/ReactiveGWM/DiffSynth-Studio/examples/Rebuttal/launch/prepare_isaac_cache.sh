#!/usr/bin/env bash

set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUTTAL_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${REBUTTAL_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/zeqingwang/anaconda3/envs/diffsynth/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/zeqingwang/zeqingwang/datasets/Isaac_processed/final_v1}"
METADATA="${METADATA:-${DATA_ROOT}/metadata_vanilla_strategy.csv}"
CACHE_ROOT="${CACHE_ROOT:-/nfs/zeqingwang/cache/ReactiveGWM/Rebuttal/Isaac_v1_480x832x101}"
WAN_ROOT="${WAN_ROOT:-/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl}"
T5_PATH="${WAN_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"
VAE_PATH="${WAN_ROOT}/Wan2.2_VAE.pth"
MODEL_PATHS="[\"${T5_PATH}\",\"${VAE_PATH}\"]"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3,4,5}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "0,3,4,5" ]]; then
  echo "ERROR: Isaac cache preparation requires CUDA_VISIBLE_DEVICES=0,3,4,5" >&2
  exit 1
fi
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
if [[ "${WORKERS_PER_GPU}" -lt 1 ]]; then
  echo "ERROR: WORKERS_PER_GPU must be positive" >&2
  exit 1
fi
WORLD_SIZE=$(("${#GPUS[@]}" * WORKERS_PER_GPU))

cd "${REPO_ROOT}"
mkdir -p "${CACHE_ROOT}/video" "${CACHE_ROOT}/first_frame" "${CACHE_ROOT}/t5"

worker_pids=()
for ((rank = 0; rank < WORLD_SIZE; rank++)); do
  gpu_index=$((rank / WORKERS_PER_GPU))
  gpu="${GPUS[$gpu_index]}"
  log="${CACHE_ROOT}/cache-rank-${rank}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" examples/Rebuttal/precompute_cache_worker.py \
    --variant isaac_v1 \
    --metadata "${METADATA}" \
    --dataset_base "${DATA_ROOT}" \
    --cache_root "${CACHE_ROOT}" \
    --model_paths "${MODEL_PATHS}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --rank "${rank}" \
    --world_size "${WORLD_SIZE}" \
    --height 480 \
    --width 832 \
    --num_frames 101 \
    --skip_existing \
    >"${log}" 2>&1 &
  worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "ERROR: Isaac cache worker failed; inspect ${CACHE_ROOT}/cache-rank-*.log" >&2
  exit 1
fi

"${PYTHON_BIN}" examples/Rebuttal/cache_layout.py finalize \
  --cache_base "${CACHE_ROOT}" \
  --variant isaac_v1 \
  --metadata "${METADATA}" \
  --dataset_base "${DATA_ROOT}" \
  --vae_path "${VAE_PATH}" \
  --t5_path "${T5_PATH}" \
  --height 480 \
  --width 832 \
  --num_frames 101 \
  >"${CACHE_ROOT}/finalize_manifest.log"

echo "Isaac cache finalized: ${CACHE_ROOT}/manifest.json"
