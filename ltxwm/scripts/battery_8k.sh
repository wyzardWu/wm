#!/bin/bash
# 通用 held-out battery: battery_8k.sh <step> <cfg> [gpu]
# 等 runs/abot8k 对应 step 的 LoRA 出现后跑 9 支 probe(2 clip × WASD + switch)
set -u
STEP=$1; CFG=$2; GPU=${3:-5}
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k/checkpoints/lora_weights_step_$(printf %05d $STEP).safetensors
TAG=s${STEP}cfg${CFG}

until [ -f "$CKPT" ]; do sleep 300; done
sleep 30
run_probe() {
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=$GPU $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/test_heldout4.jsonl --precomp $LTX/data/test_heldout4_precomp \
    --clip_idx "$1" --probe "$2" --lora $CKPT --frames 121 --cfg $CFG \
    --out /data/yuzhewu/vrisingroam/ltx_probes/8k_${TAG}_ho$1_$2.mp4 \
    > $LTX/probe_8k_${TAG}_ho$1_$2.log 2>&1
  echo "[battery $TAG] done clip$1 $2 $(date)"
}
for idx in 0 1; do for p in W_hold A_hold S_hold D_hold; do run_probe $idx $p; done; done
run_probe 0 switch_WS
echo "BATTERY_8K_${TAG}_DONE $(date)"
