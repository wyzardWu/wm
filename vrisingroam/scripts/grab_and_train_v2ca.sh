#!/bin/bash
# Card-grabber + trainer + keeper for v2_ca_bidir (runs on node2 via nohup).
# Waits until >=4 GPUs are stably free (<5GB used AND <10% util for 2 checks
# 5 min apart), then launches the CA bidirectional retrain on the first 4 free
# cards and keeps it alive to the 30k budget (weights-only resume pattern).
cd "$(dirname "$0")/.."
. env.sh
TABLE=/data/yuzhewu/vrisingroam/processed/action_context_table_v2.pt
BUDGET=30000
LOG=runs/v2ca_grab.log
log(){ echo "$(date '+%m-%d %H:%M') $*" >> "$LOG"; }

free_gpus(){
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 5000 && $3 < 10 {print $1}'
}

wait_for_cards(){
  while :; do
    A=$(free_gpus)
    if [ "$(echo "$A" | grep -c .)" -ge 4 ]; then
      sleep 300   # stability window: still free 5 min later?
      B=$(free_gpus)
      COMMON=$(comm -12 <(echo "$A" | sort) <(echo "$B" | sort) | head -4)
      if [ "$(echo "$COMMON" | grep -c .)" -ge 4 ]; then
        echo "$COMMON" | paste -sd, -
        return
      fi
    fi
    sleep 120
  done
}

launch_ca(){  # $1 = gpus csv, $2 = resume ckpt or empty, $3 = remaining steps
  EXTRA=""
  [ -n "$2" ] && EXTRA="--resume_from_ckpt $2"
  log "launch v2_ca_bidir gpus=$1 remain=$3 ckpt=${2:-v1-teacher-34k}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=$1 NPROC=4 nohup bash scripts/train_bidirectional.sh \
    data/processed/combined_5d v2_ca_bidir \
    --dataset_metadata_path data/processed/combined_5d/metadata_train.csv \
    --action_context_table $TABLE \
    $EXTRA --max_train_steps "$3" --action_dropout_prob 0.1 \
    >> runs/v2_ca_launch.log 2>&1 < /dev/null &
}

log "grabber up; waiting for 4 free cards"
GPUS=$(wait_for_cards)
log "grabbed cards: $GPUS"
launch_ca "$GPUS" runs/v1_full_ext/step-4000.safetensors $BUDGET

MISS=0
while :; do
  N=$(ls runs/v2_ca_bidir/step-*.safetensors 2>/dev/null \
      | sed 's/.*step-//; s/\.safetensors//' | sort -n | tail -1); N=${N:-0}
  [ "$N" -ge "$BUDGET" ] && { log "v2_ca_bidir budget complete ($N)"; break; }
  if pgrep -u yuzhewu -f "bidirectional.trai[n]" >/dev/null; then MISS=0; else
    MISS=$((MISS+1))
    if [ "$MISS" -ge 2 ]; then
      REMAIN=$((BUDGET - N)); [ "$REMAIN" -lt 100 ] && REMAIN=100
      GPUS=$(wait_for_cards)   # cards may have been lost while we were down
      log "relaunch on $GPUS remain=$REMAIN"
      if [ "$N" -gt 0 ]; then
        launch_ca "$GPUS" "runs/v2_ca_bidir/step-$N.safetensors" "$REMAIN"
      else
        launch_ca "$GPUS" runs/v1_full_ext/step-4000.safetensors "$BUDGET"
      fi
      MISS=0; sleep 900
    fi
  fi
  sleep 300
done
log "GRABBER DONE: v2_ca_bidir at $BUDGET"
