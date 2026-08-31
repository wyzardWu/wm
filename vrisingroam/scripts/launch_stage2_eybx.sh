#!/bin/bash
# Stage-2 CD for EYBX scene-switch: teacher = student = stage1 causal ckpt
# (P1 same-trajectory rule), zhiyang recipe verbatim, 4 cards x accum2 = batch 8.
set -eo pipefail
cd ~/vrisingroam && . env.sh
cd ~/vrisingroam/cf_distill_3stage/stage2_cd

STEP="${STEP:-4000}"
CKPT=/data/yuzhewu/vrisingroam/distill/runs/eybx_ar_tf_v1/dits/dit_step${STEP}.safetensors
GPUS="${GPUS:-2,3,5,7}" NPROC="${NPROC:-4}" PORT="${PORT:-29594}"
DATA=/data/yuzhewu/vrisingroam/distill/data/eybx_F26_v1
OUT=/data/yuzhewu/vrisingroam/distill/runs/eybx_cd_v1
FSDP_CFG=../common/accelerate_fsdp.yaml

echo "[stage2/eybx] teacher=student=$CKPT data=$DATA -> $OUT"
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
