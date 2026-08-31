#!/bin/bash
set -u
STEP=10000; GPU=3
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k_v3/checkpoints/lora_weights_step_10000.safetensors
TAB=$LTX/tables/ltx_action_tables_v31.pt
run() {
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=$GPU $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/ood3.jsonl --precomp $LTX/data/ood3_precomp --clip_idx 0 \
    --probe $1 --lora $CKPT --tables $TAB \
    --frames 121 --cfg 2.0 --out /data/yuzhewu/vrisingroam/ltx_probes/v3_s${STEP}_ood1_$1.mp4 \
    > $LTX/probe_v3_s${STEP}_ood1_$1.log 2>&1
  echo "[v3 ood1 s$STEP] done $1 $(date)"
}
run W_J
run A_L
run D_I
echo "V3_OOD1_S10000_DONE $(date)"
