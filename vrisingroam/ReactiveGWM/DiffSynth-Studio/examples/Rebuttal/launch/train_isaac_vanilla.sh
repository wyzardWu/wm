#!/usr/bin/env bash

set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUTTAL_DIR="$(cd "${LAUNCH_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${REBUTTAL_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/zeqingwang/anaconda3/envs/diffsynth/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/zeqingwang/anaconda3/envs/diffsynth/bin/accelerate}"
DATA_ROOT="${DATA_ROOT:-/home/zeqingwang/zeqingwang/datasets/Isaac_processed/final_v1}"
METADATA="${METADATA:-${DATA_ROOT}/metadata_vanilla_strategy.csv}"
CACHE_ROOT="${CACHE_ROOT:-/nfs/zeqingwang/cache/ReactiveGWM/Rebuttal/Isaac_v1_480x832x101}"
WAN_ROOT="${WAN_ROOT:-/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl}"
OUT="${OUT:-/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/Isaac_v1_vanilla_strategy}"
PORT="${PORT:-29742}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3,4,5}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "0,3,4,5" ]]; then
  echo "ERROR: Isaac training requires CUDA_VISIBLE_DEVICES=0,3,4,5" >&2
  exit 1
fi
if [[ ! -f "${CACHE_ROOT}/manifest.json" ]]; then
  echo "ERROR: validated Isaac cache manifest is missing: ${CACHE_ROOT}/manifest.json" >&2
  exit 1
fi

cd "${REPO_ROOT}"
PYTHONUNBUFFERED=1 "${ACCELERATE_BIN}" launch \
  --num_processes=4 \
  --multi_gpu \
  --main_process_port="${PORT}" \
  examples/Rebuttal/train.py \
  --variant isaac_v1 \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${METADATA}" \
  --output_path "${OUT}" \
  --wan_root "${WAN_ROOT}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --use_cached_dataset \
  --cache_root "${CACHE_ROOT}" \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 30000 \
  --save_steps 1000 \
  --prompt_dropout_prob 0.1 \
  --action_dropout_prob 0.0 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --height 480 \
  --width 832 \
  --num_frames 101 \
  --action_hold_window 1 \
  --expected_num_processes 4 \
  "$@"
