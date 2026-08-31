#!/bin/bash
# Keep GPUs 0-3 productively occupied at all times (runs on node2 via nohup):
#   - if the training process vanishes for 2 consecutive checks (>=10 min,
#     long enough to never race the orchestrator's stop->relaunch window),
#     relaunch from the latest checkpoint with the right dataset;
#   - once the global 30k-step budget is done, keep one GPU busy with a
#     rolling GT-replay eval loop instead of going idle.
cd "$(dirname "$0")/.."
log() { echo "$(date +%m-%d\ %H:%M) $*"; }

latest_step() {  # $1 = run dir
  ls "$1"/step-*.safetensors 2>/dev/null | sed 's/.*step-//; s/\.safetensors//' | sort -n | tail -1
}

MISS=0
log "keeper up"
while :; do
  if pgrep -f "bidirectional.trai[n]" >/dev/null; then
    MISS=0
  else
    MISS=$((MISS + 1))
    log "training absent (strike $MISS/2)"
    if [ "$MISS" -ge 2 ]; then
      N1=$(latest_step runs/v1_nativecrop); N1=${N1:-0}
      N2=$(latest_step runs/v1_full); N2=${N2:-0}
      N3=$(latest_step runs/v1_full_ext); N3=${N3:-0}
      TOTAL=$((N1 + N2 + N3))
      if [ "$TOTAL" -ge ${BUDGET:-30000} ]; then
        log "budget complete ($TOTAL); switching to eval loop"
        CKPT=runs/v1_full/step-$N2.safetensors
        [ -e "$CKPT" ] || CKPT=runs/v1_nativecrop/step-$N1.safetensors
        while :; do
          CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/rgwm/bin/python scripts/eval_gt.py \
            --ckpt "$CKPT" --data_root data/processed/combined_5d \
            --clips $(ls data/processed/combined_5d/clips/*/*.mp4 2>/dev/null | shuf -n 3 | sed 's|data/processed/combined_5d/||') \
            --out_dir runs/eval_rolling >> runs/eval_rolling.log 2>&1
          sleep 600
        done
      fi
      if [ -f data/processed/combined_5d/metadata.csv ] && [ "$N1" -gt 0 ]; then
        DATA=data/processed/combined_5d; RUN=${RUN_NAME:-v1_full}
        if [ "$N3" -gt 0 ]; then CKPT=runs/v1_full_ext/step-$N3.safetensors
        elif [ "$N2" -gt 0 ]; then CKPT=runs/v1_full/step-$N2.safetensors
        else CKPT=runs/v1_nativecrop/step-$N1.safetensors; fi
      else
        DATA=data/processed/20260731_nativecrop; RUN=v1_nativecrop
        if [ "$N1" -gt 0 ]; then CKPT=runs/v1_nativecrop/step-$N1.safetensors; else CKPT=""; fi
      fi
      REMAIN=$((${BUDGET:-30000} - TOTAL)); [ "$REMAIN" -lt 100 ] && REMAIN=100
      EXTRA=""
      [ -n "$CKPT" ] && EXTRA="--resume_from_ckpt $CKPT"
      [ "$RUN" = v1_nativecrop ] && [ ! -f data/processed/combined_5d/metadata.csv ] \
        && EXTRA="$EXTRA --dataset_metadata_path $DATA/metadata_ready.csv"
      log "relaunching: data=$DATA run=$RUN remain=$REMAIN ckpt=${CKPT:-none}"
      CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} NPROC=${NP:-4} nohup bash scripts/train_bidirectional.sh \
        "$DATA" "$RUN" $EXTRA --max_train_steps "$REMAIN" \
        >> runs/${RUN}_keeper_relaunch.log 2>&1 &
      MISS=0
      sleep 600
    fi
  fi
  sleep 300
done
