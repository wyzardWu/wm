#!/bin/bash
set -u
STEP=10000; GPU=3
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k_v3/checkpoints/lora_weights_step_10000.safetensors
TAB=$LTX/tables/ltx_action_tables_v31.pt
run() {
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=$GPU $PY $LTX/ltxwm/probe_action.py \
    --manifest $1 --precomp $2 --clip_idx $3 --probe $4 --lora $CKPT --tables $TAB \
    --frames 121 --cfg 2.0 --out /data/yuzhewu/vrisingroam/ltx_probes/v3_s${STEP}_$5.mp4 \
    > $LTX/probe_v3_s${STEP}_$5.log 2>&1
  echo "[v3 ood s$STEP] done $5 $(date)"
}
OOD="$LTX/data/ood3.jsonl $LTX/data/ood3_precomp"
run $OOD 2 D_I ood3_D_I
run $OOD 2 I_hold ood3_I_hold
run $OOD 1 A_L ood2_A_L
echo "V3_OOD_S10000_DONE $(date)"
