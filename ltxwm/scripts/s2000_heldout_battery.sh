#!/bin/bash
# 等 abot20k step-2000 checkpoint + 测试集 precompute,然后 held-out probe battery (GPU 5)
# 测试集 = train2000 之外的 4 条 ABot episode (data/test_heldout4.jsonl, seed=42 抽样)
set -u
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT_SRC=$LTX/runs/abot20k/checkpoints/lora_weights_step_02000.safetensors
CKPT=$LTX/runs/abot20k/lora_step02000_keep.safetensors
OUT=/data/yuzhewu/vrisingroam/ltx_probes

until [ -f "$CKPT_SRC" ]; do sleep 120; done
sleep 30
cp "$CKPT_SRC" "$CKPT"
echo "[battery] ckpt copied $(date)"

until [ "$(find $LTX/data/test_heldout4_precomp/conditions -name '*.pt' 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
echo "[battery] test precomp ready $(date)"

run_probe() {
  local idx=$1 p=$2
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=5 $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/test_heldout4.jsonl --precomp $LTX/data/test_heldout4_precomp \
    --clip_idx "$idx" --probe "$p" --lora $CKPT --frames 121 \
    --out $OUT/s2000_heldout${idx}_${p}.mp4 > $LTX/probe_s2000_heldout${idx}_${p}.log 2>&1
  echo "[battery] done clip$idx $p $(date)"
}

for idx in 0 1; do
  for p in W_hold A_hold S_hold D_hold; do
    run_probe $idx $p
  done
done
run_probe 0 switch_WS
echo "S2000_HELDOUT_BATTERY_DONE $(date)"
