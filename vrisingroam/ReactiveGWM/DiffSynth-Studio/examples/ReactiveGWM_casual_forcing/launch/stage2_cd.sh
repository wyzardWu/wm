#!/usr/bin/env bash
# Stage 2 CD launch — multi-GPU accelerate DDP (OOM 切 FSDP 单 unit).
#
# Env overrides:
#   BASE_CKPT      override cfg.base_ckpt
#   DATA_ROOT      override cfg.dataset_base
#   OUT            override cfg.output_path
#   STUDENT_INIT   override cfg.student_init (generator + EMA 起点)
#   TEACHER_CKPT   override cfg.teacher_ckpt (默认 = student_init)
#   RESUME_STATE   path to a `state-<N>/` dir (含 ema.safetensors) to resume from
#   CFG            override config yaml (默认 configs/stage2_cd.yaml)
#   NUM_PROCESSES  进程数 (默认 8; 4-card 用 4)
#   MAIN_PORT      主进程端口 (默认 29521; 与 Stage 1 的 29520 错开, 可与 Stage 1 同时跑)
#
# 4-card 正式 run (GPU 0-3, 与 Stage 1 GPU 4-7 并行; 默认 CFG 已是 canonical stage2_cd.yaml):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
#     bash examples/ReactiveGWM_casual_forcing/launch/stage2_cd.sh

set -euo pipefail

# 共享 path/env setup + cf_build_args (见 _common.sh)。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CFG="${CFG:-${HERE}/configs/stage2_cd.yaml}"

ARGS=()
cf_build_args cd "${CFG}"

accelerate launch \
    --num_processes "${NUM_PROCESSES:-8}" \
    --num_machines 1 \
    --multi_gpu \
    --mixed_precision bf16 \
    --main_process_port "${MAIN_PORT:-29521}" \
    "${TRAIN_PY}" "${ARGS[@]}"
