#!/bin/bash
# Stage-1 causalize for EYBX scene-switch (distillation feasibility run).
# Init = the eybx bidirectional overfit ckpt (825-key, table-mode); data =
# converted gm clips with combined action⊕scene 32-token context.
# 4 GPUs (2,3,5,7) x accum2 = effective batch 8 (zhiyang parity; user 8/25:
# once arbitrary-time switching lands, bidirectional stops and all four cards
# go to distillation).
set -eo pipefail
cd ~/vrisingroam && . env.sh
cd ~/vrisingroam/cf_distill_3stage/stage1_ar_teacher

INIT="${INIT:?set INIT=/path/to/eybx_bidir_ckpt.safetensors}"
GPUS="${GPUS:-2,3,5,7}" NPROC="${NPROC:-4}" PORT="${PORT:-29593}"
DATA="${DATA:-/data/yuzhewu/vrisingroam/distill/data/eybx_F26_v1}"
OUT="${OUT:-/data/yuzhewu/vrisingroam/distill/runs/eybx_ar_tf_v1}"

CONDA_ENV="" GPUS=$GPUS NPROC=$NPROC PORT=$PORT \
DATA=$DATA INIT=$INIT OUT=$OUT \
bash run.sh --grad_accum 2 "$@"
