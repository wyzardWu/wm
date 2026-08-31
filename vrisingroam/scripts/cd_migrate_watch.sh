#!/bin/bash
# Watch 252: when >=1 extra GPU has >=105G free (besides GPU7 where CD-v2 runs single-card),
# migrate CD-v2 to a multi-card run (resume auto, batch kept at 8 via ACCUM=8/NPROC).
cd ~/vrisingroam && . env.sh
RUN=/data/yuzhewu/vrisingroam/distill/runs/gm_cd_v2
LOG=$RUN/migrate_watch.log
NEED=105000
while true; do
  CANDS=()
  for g in 0 1 2 3 4 5 6; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    [ "$free" -ge $NEED ] && CANDS+=($g)
  done
  # only migrate once a CD state exists to resume from
  HAVE_STATE=$([ -f $RUN/progress.json ] && echo 1 || echo 0)
  if [ "${#CANDS[@]}" -ge 1 ] && [ "$HAVE_STATE" -ge 1 ]; then
    # pick up to 3 extra cards + GPU7 = up to 4 ranks; accum must divide 8 -> use 2 or 4 ranks
    N=$(( ${#CANDS[@]} + 1 )); [ $N -ge 4 ] && N=4; [ $N -eq 3 ] && N=2
    EXTRA=("${CANDS[@]:0:$((N-1))}")
    GPUS=$(IFS=,; echo "${EXTRA[*]}"),7
    ACCUM=$((8 / N))
    echo "$(date '+%m-%d %H:%M') migrating CD-v2 -> GPUS=$GPUS NPROC=$N ACCUM=$ACCUM" >> $LOG
    pkill -f "cd_g[m].py"; sleep 8
    FLAG=/data/yuzhewu/vrisingroam/distill/cdmig_starting; rm -f $FLAG
    for g in "${EXTRA[@]}" 7; do nohup python scripts/hold_feed.py $g $FLAG >> $RUN/hold_mig.log 2>&1 & done
    sleep 20; touch $FLAG
    DIT_STEP=12000 GPUS=$GPUS NPROC=$N ACCUM=$ACCUM PORT=29592 nohup bash scripts/launch_stage2_252.sh --resume 1 > $RUN/../stage2v2_migrated_launch.log 2>&1 &
    echo "$(date '+%m-%d %H:%M') launched pid $!" >> $LOG
    exit 0
  fi
  sleep 120
done
