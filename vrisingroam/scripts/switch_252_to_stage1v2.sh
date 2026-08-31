#!/bin/bash
# 252 switch: stop filtered teacher -> hold GPUs 2/3/5/7 -> launch stage1-v2 (noblock0).
# Holders trim+feed as stage1-v2 ranks allocate, so the cards are never exposed.
set -eo pipefail
cd ~/vrisingroam && . env.sh
FLAG=/data/yuzhewu/vrisingroam/distill/stage1v2_starting
LOG=/data/yuzhewu/vrisingroam/distill/runs/stage1v2_switch.log
rm -f "$FLAG"
mkdir -p /data/yuzhewu/vrisingroam/distill/runs

echo "$(date +%H:%M:%S) stopping filtered teacher" | tee -a "$LOG"
pkill -f "bidirectional.trai[n]" || true
pkill -f "train_bidirectiona[l].sh" || true
sleep 2

for g in 2 3 5 7; do
  nohup python scripts/hold_feed.py "$g" "$FLAG" >> "$LOG" 2>&1 &
done
echo "$(date +%H:%M:%S) holders launched" | tee -a "$LOG"
sleep 25
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2,3,5,7 | tee -a "$LOG"

touch "$FLAG"
echo "$(date +%H:%M:%S) launching stage1-v2" | tee -a "$LOG"
INIT_STEP="${INIT_STEP:-68000}" nohup bash scripts/launch_stage1v2_252.sh \
  > /data/yuzhewu/vrisingroam/distill/runs/stage1v2_launch.log 2>&1 &
echo "$(date +%H:%M:%S) launched pid $!" | tee -a "$LOG"
