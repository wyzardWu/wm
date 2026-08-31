#!/usr/bin/env bash
# Stage 1 AR-TF launch — 8×H200 accelerate DDP (OOM 切 FSDP 单 unit).
#
# Env overrides:
#   BASE_CKPT      override cfg.base_ckpt
#   DATA_ROOT      override cfg.dataset_base
#   OUT            override cfg.output_path
#   STUDENT_INIT   override cfg.student_init (Stage 2/3 also reads this)
#   RESUME_STATE   path to a `state-<N>/` dir written by previous run
#
# 默认 DDP. 若 OOM 切 FSDP 单 unit (--use_fsdp 由 accelerate config 控制).

set -euo pipefail

# 共享 path/env setup + cf_build_args (见 _common.sh)。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CFG="${CFG:-${HERE}/configs/stage1_ar.yaml}"

ARGS=()
cf_build_args ar_tf "${CFG}"

# Multi-GPU DDP. Defaults to 8 processes; override with NUM_PROCESSES
# (e.g. NUM_PROCESSES=4 + CUDA_VISIBLE_DEVICES=4,5,6,7 to run on GPUs 4-7).
accelerate launch \
    --num_processes "${NUM_PROCESSES:-8}" \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PORT:-29520}" \
    "${TRAIN_PY}" "${ARGS[@]}"
