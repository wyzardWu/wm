#!/bin/bash
# 252 final switch: retire filtered teacher -> hold GPUs 2/3/5/7 -> launch stage2 CD
# (teacher=student=v1 dit_step${DIT_STEP}, data=noblock0). Cards never exposed.
set -eo pipefail
cd ~/vrisingroam && . env.sh
FLAG=/data/yuzhewu/vrisingroam/distill/stage2_starting
LOG=/data/yuzhewu/vrisingroam/distill/runs/stage2_switch.log
rm -f "$FLAG"
mkdir -p /data/yuzhewu/vrisingroam/distill/runs

echo "$(date +%H:%M:%S) retiring filtered teacher" | tee -a "$LOG"
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
echo "$(date +%H:%M:%S) launching stage2 CD (DIT_STEP=${DIT_STEP:-8000})" | tee -a "$LOG"
DIT_STEP="${DIT_STEP:-8000}" nohup bash scripts/launch_stage2_252.sh \
  > /data/yuzhewu/vrisingroam/distill/runs/stage2_launch.log 2>&1 &
echo "$(date +%H:%M:%S) launched pid $!" | tee -a "$LOG"
