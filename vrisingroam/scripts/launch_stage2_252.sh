#!/bin/bash
# Stage-2 CD on 252's 4 GPUs (zhiyang-native 4-card x accum2 = effective batch 8).
# P1 route: causal AR-diffusion dit_step{N} as BOTH teacher and student init.
# Prereq at switch time: rsync the chosen dit ckpt from node2 into $CKPT below.
set -eo pipefail
cd ~/vrisingroam && . env.sh
cd ~/vrisingroam/cf_distill_3stage/stage2_cd

DIT_STEP="${DIT_STEP:-10000}"
CKPT=/data/yuzhewu/vrisingroam/distill/dits/dit_step${DIT_STEP}.safetensors
GPUS="${GPUS:-2,3,5,7}" NPROC="${NPROC:-4}" PORT="${PORT:-29585}"
DATA=/data/yuzhewu/vrisingroam/distill/data/vrising_F26_v3_noblock0
OUT=/data/yuzhewu/vrisingroam/distill/runs/gm_cd_v2
FSDP_CFG=../common/accelerate_fsdp.yaml

echo "[stage2/252] teacher=student=$CKPT data=$DATA -> $OUT"
CONDA_ENV="" CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=16 \
  accelerate launch --config_file "$FSDP_CFG" --num_processes "$NPROC" --main_process_port "$PORT" \
  cd_gm.py \
    --data "$DATA" \
    --teacher_ckpt "$CKPT" --student_ckpt "$CKPT" \
    --teacher_causal 1 --teacher_forcing \
    --out "$OUT" \
    --lr 2e-6 --grad_accum "${ACCUM:-2}" --max_steps 999000 --warmup 100 \
    --opt8bit 1 --save_every 500 --keep_last_states 2 "$@"
