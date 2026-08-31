#!/bin/bash
set -u
LTX=/data/yuzhewu/ltxwm
PY=/home/yuzhewu/miniconda3/envs/rgwm/bin/python
run() { # run <idx> <probe> <name>
  TERM=dumb TMPDIR=/data/yuzhewu/tmp CUDA_VISIBLE_DEVICES=6 $PY $LTX/ltxwm/probe_action.py \
    --manifest $LTX/data/ood3_promptonly.jsonl --precomp $LTX/data/ood3_precomp \
    --clip_idx $1 --probe $2 --baseline \
    --frames 121 --cfg 2.0 --out /data/yuzhewu/vrisingroam/ltx_probes/promptonly_$3.mp4 \
    > $LTX/probe_promptonly_$3.log 2>&1
  echo "[promptonly] done $3 $(date)"
}
run 0 W_J ood1_W_J
run 1 A_L ood2_A_L
run 2 D_I ood3_D_I
echo "PROMPTONLY_DONE $(date)"
