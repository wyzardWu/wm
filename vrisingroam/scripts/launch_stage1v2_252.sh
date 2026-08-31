#!/bin/bash
# Stage-1 v2 on 252's 4 GPUs: causalize the FILTERED (noblock) bidirectional teacher
# with the FILTERED converted clips. zhiyang-native 4-card x accum2 (= his run.sh).
set -eo pipefail
cd ~/vrisingroam && . env.sh
cd ~/vrisingroam/cf_distill_3stage/stage1_ar_teacher

INIT_STEP="${INIT_STEP:-68000}"
INIT=/data/yuzhewu/vrisingroam/teacher252/archive/noblock_step-${INIT_STEP}.safetensors
GPUS="${GPUS:-2,3,5,7}" NPROC="${NPROC:-4}" PORT="${PORT:-29586}"
DATA=/data/yuzhewu/vrisingroam/distill/data/vrising_F26_v3_noblock0
OUT=/data/yuzhewu/vrisingroam/distill/runs/gm_ar_tf_v2

CONDA_ENV="" GPUS=$GPUS NPROC=$NPROC PORT=$PORT \
DATA=$DATA INIT=$INIT OUT=$OUT \
bash run.sh --grad_accum 2 "$@"
