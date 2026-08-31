#!/bin/bash
set -u
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
CKPT=$LTX/runs/abot8k_v3/checkpoints/lora_weights_step_10000.safetensors
# 等 ood2 四宫格链结束再占 GPU 1
until grep -aq 'QUAD_DONE' $LTX/quad_ood2.log 2>/dev/null; do sleep 60; done
for p in W_camJ S_camJ A_camJ D_camJ; do
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=1 $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/firstperson.jsonl --precomp $LTX/data/firstperson_precomp --clip_idx 0 \
    --probe $p --lora $CKPT --tables $LTX/tables/ltx_action_tables_v31.pt \
    --frames 121 --cfg 2.0 --out /data/yuzhewu/vrisingroam/ltx_probes/v3_s10000_fp1_$p.mp4 \
    > $LTX/probe_v3_s10000_fp1_$p.log 2>&1
  echo "[quad_fp] done $p $(date)"
done
echo "QUAD_FP_DONE $(date)"
