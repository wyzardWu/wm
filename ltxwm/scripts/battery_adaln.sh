#!/bin/bash
# AdaLN 臂 battery: 等 3k checkpoint(lora+embedder) → held-out 8科 + OOD 3科, GPU 1
set -u
STEP=${1:-3000}; GPU=${2:-1}
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CK=$LTX/runs/abot8k_adaln/checkpoints
LORA=$CK/lora_weights_step_0$STEP.safetensors
EMB=$CK/action_adaln_step_0$STEP.pt
until [ -f "$LORA" ] && [ -f "$EMB" ]; do sleep 120; done
sleep 30
run() { # run <manifest> <precomp> <idx> <probe> <name>
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=$GPU $PY $LTX/ltxwm/probe_action.py \
    --manifest $1 --precomp $2 --clip_idx $3 --probe $4 --inject adaln \
    --lora $LORA --adaln_ckpt $EMB \
    --frames 121 --cfg 2.0 --out /data/yuzhewu/vrisingroam/ltx_probes/adaln_s${STEP}_$5.mp4 \
    > $LTX/probe_adaln_s${STEP}_$5.log 2>&1
  echo "[adaln battery s$STEP] done $5 $(date)"
}
HO="$LTX/data/test_heldout4.jsonl $LTX/data/test_heldout4_precomp"
OOD="$LTX/data/ood3.jsonl $LTX/data/ood3_precomp"
for idx in 0 1; do for p in W_hold A_hold S_hold D_hold; do run $HO $idx $p ho${idx}_${p}; done; done
run $OOD 2 D_I ood3_D_I
run $OOD 2 I_hold ood3_I_hold
run $OOD 1 A_L ood2_A_L
echo "ADALN_BATTERY_S${STEP}_DONE $(date)"
