#!/usr/bin/env bash

set -euo pipefail

REBUTTAL_LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUTTAL_DIR="$(cd "${REBUTTAL_LAUNCH_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${REBUTTAL_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/zeqingwang/anaconda3/envs/reactivegwm/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/zeqingwang/anaconda3/envs/reactivegwm/bin/accelerate}"

DATA_ROOT="${DATA_ROOT:-/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF2}"
WAN_ROOT="${WAN_ROOT:-/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl}"
CACHE_BASE="${CACHE_BASE:-/nfs/zeqingwang/cache/ReactiveGWM/Rebuttal/SF2_480x608x101}"
OUTPUT_BASE="${OUTPUT_BASE:-/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
IFS=',' read -r -a REBUTTAL_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#REBUTTAL_GPUS[@]}"
FORMAL_NUM_GPUS=6

T5_PATH="${WAN_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"
VAE_PATH="${WAN_ROOT}/Wan2.2_VAE.pth"
CACHE_MODEL_PATHS="[\"${T5_PATH}\",\"${VAE_PATH}\"]"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file is missing: $1" >&2
    exit 1
  fi
}

require_directory() {
  if [[ ! -d "$1" ]]; then
    echo "ERROR: required directory is missing: $1" >&2
    exit 1
  fi
}

validate_common_environment() {
  require_file "${PYTHON_BIN}"
  require_file "${ACCELERATE_BIN}"
  require_directory "${DATA_ROOT}"
  require_directory "${WAN_ROOT}"
  require_directory "${TOKENIZER_PATH}"
  require_file "${T5_PATH}"
  require_file "${VAE_PATH}"
}

launch_formal_training() {
  local variant="$1"
  local metadata="$2"
  local cache_root="$3"
  local output_path="$4"
  local port="$5"
  shift 5
  local formal_num_gpus="${FORMAL_NUM_GPUS}"
  if [[ "${variant}" == "v2" ]]; then
    formal_num_gpus=4
  fi

  validate_common_environment
  if [[ "${NUM_GPUS}" -ne "${formal_num_gpus}" ]]; then
    echo "ERROR: formal ${variant} training requires exactly ${formal_num_gpus} visible GPUs; got ${NUM_GPUS}: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
  require_file "${metadata}"
  require_file "${cache_root}/manifest.json"

  cd "${REPO_ROOT}"
  echo "variant=${variant}"
  echo "metadata=${metadata}"
  echo "cache=${cache_root}"
  echo "output=${output_path}"
  echo "GPUs=${CUDA_VISIBLE_DEVICES}"

  PYTHONUNBUFFERED=1 "${ACCELERATE_BIN}" launch \
    --num_processes="${formal_num_gpus}" \
    --multi_gpu \
    --main_process_port="${port}" \
    examples/Rebuttal/train.py \
    --variant "${variant}" \
    --dataset_base_path "${DATA_ROOT}" \
    --dataset_metadata_path "${metadata}" \
    --output_path "${output_path}" \
    --wan_root "${WAN_ROOT}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --use_cached_dataset \
    --cache_root "${cache_root}" \
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
    --width 608 \
    --num_frames 101 \
    --action_hold_window 10 \
    --expected_num_processes "${formal_num_gpus}" \
    "$@"
}
