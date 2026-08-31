#!/bin/bash
set -u
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k/checkpoints/lora_weights_step_02000.safetensors
for cfg in 1.0 2.0; do
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=5 $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/test_heldout4.jsonl --precomp $LTX/data/test_heldout4_precomp \
    --clip_idx 0 --probe W_hold --lora $CKPT --frames 121 --cfg $cfg \
    --out /data/yuzhewu/vrisingroam/ltx_probes/8k_s2000_ho0_W_cfg${cfg}.mp4 \
    > $LTX/probe_8k_s2000_W_cfg${cfg}.log 2>&1
  echo "done cfg$cfg $(date)"
done
echo CFG_SWEEP_DONE
