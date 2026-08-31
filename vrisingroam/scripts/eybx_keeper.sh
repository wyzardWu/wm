#!/bin/bash
# EYBX overfit keeper: keep the trainer alive on GPUs 2,3 indefinitely
# (user directive 2026-08-24: no step cap, hold both cards).
# Relaunches from the latest full state dir (--resume_state) on crash/exit.
RUN=~/vrisingroam/runs/eybx_overfit_v3
LOG=/data/yuzhewu/eybxroam/keeper.log
while true; do
  if ! pgrep -f "bidirectional.train" >/dev/null; then
    STATE=$(ls -dt "$RUN"/*state*/ 2>/dev/null | head -1)
    EXTRA="--max_train_steps 999000"
    [ -n "$STATE" ] && EXTRA="$EXTRA --resume_state ${STATE%/}"
    echo "[$(date '+%F %T')] trainer down, relaunching (resume=${STATE:-none})" >> "$LOG"
    cd ~/vrisingroam && nohup bash scripts/train_eybx_overfit.sh eybx_overfit_v3 $EXTRA \
      >> /data/yuzhewu/eybxroam/train_v3.log 2>&1 &
    sleep 600   # give it time to load before re-checking
  fi
  sleep 60
done
