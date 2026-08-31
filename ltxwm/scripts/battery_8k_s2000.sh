#!/bin/bash
# abot8k step-2000 held-out probe battery (GPU 5, 与训练共卡)
set -u
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k/checkpoints/lora_weights_step_02000.safetensors
OUT=/data/yuzhewu/vrisingroam/ltx_probes

run_probe() {
  local idx=$1 p=$2
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=5 $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/test_heldout4.jsonl --precomp $LTX/data/test_heldout4_precomp \
    --clip_idx "$idx" --probe "$p" --lora $CKPT --frames 121 \
    --out $OUT/8k_s2000_ho${idx}_${p}.mp4 > $LTX/probe_8k_s2000_ho${idx}_${p}.log 2>&1
  echo "[battery] done clip$idx $p $(date)"
}

for idx in 0 1; do
  for p in W_hold A_hold S_hold D_hold; do
    run_probe $idx $p
  done
done
run_probe 0 switch_WS
echo "BATTERY_8K_S2000_DONE $(date)"
