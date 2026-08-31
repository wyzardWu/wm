#!/bin/bash
set -u
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k_v3/checkpoints/lora_weights_step_10000.safetensors
for p in W_camJ S_camJ A_camJ D_camJ; do
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=1 $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/ood3.jsonl --precomp $LTX/data/ood3_precomp --clip_idx 1 \
    --probe $p --lora $CKPT --tables $LTX/tables/ltx_action_tables_v31.pt \
    --frames 121 --cfg 2.0 --out /data/yuzhewu/vrisingroam/ltx_probes/v3_s10000_ood2_$p.mp4 \
    > $LTX/probe_v3_s10000_ood2_$p.log 2>&1
  echo "[quad] done $p $(date)"
done
echo "QUAD_DONE $(date)"
