#!/bin/bash
# v3 梯子: battery_v3.sh <step> <gpu>  — held-out 9支 + OOD D_I/I_hold 判决科目, cfg2, v3 表
set -u
STEP=$1; GPU=${2:-5}
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k_v3/checkpoints/lora_weights_step_$(printf %05d $STEP).safetensors
TAB=$LTX/tables/ltx_action_tables_v31.pt
until [ -f "$CKPT" ]; do sleep 180; done
sleep 30
run() { # run <manifest> <precomp> <idx> <probe> <name>
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=$GPU $PY $LTX/ltxwm/probe_action.py \
    --manifest $1 --precomp $2 --clip_idx $3 --probe $4 --lora $CKPT --tables $TAB \
    --frames 121 --cfg 2.0 --out /data/yuzhewu/vrisingroam/ltx_probes/v3_s${STEP}_$5.mp4 \
    > $LTX/probe_v3_s${STEP}_$5.log 2>&1
  echo "[v3 battery s$STEP] done $5 $(date)"
}
HO="$LTX/data/test_heldout4.jsonl $LTX/data/test_heldout4_precomp"
OOD="$LTX/data/ood3.jsonl $LTX/data/ood3_precomp"
for idx in 0 1; do for p in W_hold A_hold S_hold D_hold; do run $HO $idx $p ho${idx}_${p}; done; done
run $OOD 2 D_I ood3_D_I
run $OOD 2 I_hold ood3_I_hold
run $OOD 1 A_L ood2_A_L
echo "V3_BATTERY_S${STEP}_DONE $(date)"
