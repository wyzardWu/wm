#!/bin/bash
# Keeper for CD-v2 on 252 GPUs 2,3: if the training dies (OOM/crash), hold cards and resume.
cd ~/vrisingroam && . env.sh
RUN=/data/yuzhewu/vrisingroam/distill/runs/gm_cd_v2; LOG=$RUN/keeper.log
while true; do
  sleep 180
  if pgrep -f "cd_g[m].py" >/dev/null || pgrep -f "launch_stage2_25[2]" >/dev/null || pgrep -f "hold_fee[d].py" >/dev/null; then continue; fi
  grep -q "CD-INIT DONE" $RUN/train.log 2>/dev/null && { echo "$(date '+%m-%d %H:%M') CD done, keeper exit" >> $LOG; exit 0; }
  echo "$(date '+%m-%d %H:%M') CD not running -> re-holding 2,3 and resuming" >> $LOG
  FLAG=/data/yuzhewu/vrisingroam/distill/cdkeep_starting; rm -f $FLAG
  for g in 2 3; do nohup python scripts/hold_feed.py $g $FLAG >> $RUN/hold_keeper.log 2>&1 & done
  sleep 20; touch $FLAG
  DIT_STEP=12000 GPUS=2,3 NPROC=2 ACCUM=4 PORT=29595 nohup bash scripts/launch_stage2_252.sh --resume 1 >> $RUN/../stage2v2_keeper_launch.log 2>&1 &
  sleep 600
done
