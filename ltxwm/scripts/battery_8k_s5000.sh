#!/bin/bash
set -u
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k/checkpoints/lora_weights_step_05000.safetensors
until [ -f $CKPT ]; do sleep 300; done
sleep 30
for idx in 0 1; do for p in W_hold A_hold S_hold D_hold; do
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=5 $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/test_heldout4.jsonl --precomp $LTX/data/test_heldout4_precomp \
    --clip_idx $idx --probe $p --lora $CKPT --frames 121 \
    --out /data/yuzhewu/vrisingroam/ltx_probes/8k_s5000_ho${idx}_${p}.mp4 \
    > $LTX/probe_8k_s5000_ho${idx}_${p}.log 2>&1
  echo "done clip$idx $p $(date)"
done; done
echo BATTERY_8K_S5000_DONE
