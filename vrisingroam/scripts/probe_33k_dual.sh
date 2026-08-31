#!/bin/bash
# Runs on 252 (node1): wait for p3 micro-60000 (= true 33k) on node2, pull the
# checkpoint, run two double-window chains (W/D synthetic + gt10s GT stream)
# at cfg1, leave results in eval_v2ca/probe33k/. User reviews in the morning.
cd ~/vrisingroam
. env.sh
E=/data/yuzhewu/vrisingroam/eval_v2ca
OUT=$E/probe33k
mkdir -p "$OUT"
LOG=$OUT/run.log
echo "$(date '+%m-%d %H:%M') waiting for step-60000" >> "$LOG"

until ssh -o BatchMode=yes -o ConnectTimeout=20 yuzhewu@h200-node2 \
      'ls ~/vrisingroam/runs/v2_ca_bidir_p4/step-6000.safetensors >/dev/null 2>&1' < /dev/null; do
  sleep 600
done
rsync -a yuzhewu@h200-node2:~/vrisingroam/runs/v2_ca_bidir_p4/step-6000.safetensors \
      /data/yuzhewu/vrisingroam/eval_ckpts/ >> "$LOG" 2>&1

# pick the freer of GPU 0/1
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1<2 {print $2" "$1}' | sort -n | head -1 | cut -d" " -f2)
echo "$(date '+%m-%d %H:%M') step-60000 ready, gpu=$G" >> "$LOG"

CUDA_VISIBLE_DEVICES=$G python scripts/eval_chain.py \
  --ckpt /data/yuzhewu/vrisingroam/eval_ckpts/step-6000.safetensors \
  --image /data/yuzhewu/vrisingroam/eval_teacher/probeW_row5.png \
  --name WD_2win_33k --windows 2 --keys W/D --out_dir "$OUT" \
  --cfg 1.0 --action_cfg 1.0 \
  --action_context_table /data/yuzhewu/vrisingroam/processed/action_context_table_v3.pt \
  >> "$LOG" 2>&1

CUDA_VISIBLE_DEVICES=$G python scripts/eval_chain.py \
  --ckpt /data/yuzhewu/vrisingroam/eval_ckpts/step-6000.safetensors \
  --image "$E/gt10s_seed.png" \
  --name gt10s_33k --windows 2 --full_actions "$E/gt10s_actions.parquet" --out_dir "$OUT" \
  --cfg 1.0 --action_cfg 1.0 \
  --action_context_table /data/yuzhewu/vrisingroam/processed/action_context_table_v3.pt \
  >> "$LOG" 2>&1

CH=$(ls $OUT/gt10s_33k*chain2.mp4 2>/dev/null | grep -v hud | head -1)
[ -n "$CH" ] && ffmpeg -y -loglevel error -i "$CH" -i "$E/gt10s_GT.mp4" \
  -filter_complex '[0:v][1:v]hstack[v]' -map '[v]' "$OUT/gt10s_33k_SBS.mp4" >> "$LOG" 2>&1
echo "$(date '+%m-%d %H:%M') PROBE33K_DONE" >> "$LOG"
