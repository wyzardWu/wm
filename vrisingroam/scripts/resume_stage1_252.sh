#!/bin/bash
# Resume stage1-v1 (state_step8000, migrated from node2) on 252 GPUs 2,3 with noblock0 data.
set -eo pipefail
cd ~/vrisingroam && . env.sh
cd ~/vrisingroam/cf_distill_3stage/stage1_ar_teacher
CONDA_ENV="" GPUS="${GPUS:-2,3}" NPROC=2 PORT=29590 \
DATA=/data/yuzhewu/vrisingroam/distill/data/vrising_F26_v3_noblock0 \
INIT=/data/yuzhewu/vrisingroam/eval_ckpts/step-65000.safetensors \
OUT=/data/yuzhewu/vrisingroam/distill/runs/gm_ar_tf_v1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash run.sh --grad_accum 4 "$@"
