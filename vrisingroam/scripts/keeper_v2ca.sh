#!/bin/bash
# Keeper for v2_ca_bidir on node2: weights-only resume + remaining-budget
# pattern, re-grabs 4 stably-free cards on relaunch (initial run was started
# manually on 1,2,4,5). Budget 30000.
cd "$(dirname "$0")/.."
. env.sh
TABLE=/data/yuzhewu/vrisingroam/processed/action_context_table_v2.pt
BUDGET=60000
LOG=runs/v2ca_grab.log
log(){ echo "$(date '+%m-%d %H:%M') $*" >> "$LOG"; }

free_gpus(){
  # <20GB used AND <10% util: tolerates small inference residents (e.g. 13GB)
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 20000 && $3 < 10 {print $1}'
}
wait_for_cards(){  # returns 4 cards if possible, else settles for 2 stable ones
  while :; do
    A=$(free_gpus)
    N_A=$(echo "$A" | grep -c .)
    if [ "$N_A" -ge 2 ]; then
      sleep 300
      B=$(free_gpus)
      COMMON=$(comm -12 <(echo "$A" | sort) <(echo "$B" | sort))
      N_C=$(echo "$COMMON" | grep -c .)
      if [ "$N_C" -ge 4 ]; then echo "$COMMON" | head -4 | paste -sd, -; return; fi
      if [ "$N_C" -ge 2 ]; then echo "$COMMON" | head -2 | paste -sd, -; return; fi
    fi
    sleep 120
  done
}
launch_ca(){  # $1 gpus csv, $2 resume ckpt ("" = cold start), $3 remaining steps
  EXTRA=""
  [ -n "$2" ] && EXTRA="--resume_from_ckpt $2"
  NP=$(echo "$1" | awk -F, "{print NF}")
  ACC=$((4 / NP)); [ "$ACC" -lt 1 ] && ACC=1   # keep effective batch 4 (v1 recipe)
  log "relaunch v2_ca_bidir gpus=$1 np=$NP accum=$ACC remain=$3 ckpt=${2:-cold-wan-base}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=$1 NPROC=$NP nohup bash scripts/train_bidirectional.sh \
    data/processed/combined_5d v2_ca_bidir \
    --dataset_metadata_path data/processed/combined_5d/metadata_train.csv \
    --action_context_table $TABLE \
    $EXTRA --max_train_steps "$3" --action_dropout_prob 0.1 \
    --gradient_accumulation_steps $ACC \
    >> runs/v2_ca_launch.log 2>&1 < /dev/null &
}

log "keeper_v2ca up"
MISS=0
while :; do
  N=$(ls runs/v2_ca_bidir/step-*.safetensors 2>/dev/null \
      | sed 's/.*step-//; s/\.safetensors//' | sort -n | tail -1); N=${N:-0}
  [ "$N" -ge "$BUDGET" ] && { log "v2_ca_bidir budget complete ($N)"; break; }
  if pgrep -u yuzhewu -f "bidirectional.trai[n]" >/dev/null; then
    MISS=0
    # upgrade path: if we are running on 2 cards but 4+ others are stably free, restart on 4
    CUR_NP=$(pgrep -u yuzhewu -f "bidirectional.trai[n]" | wc -l)
    if [ "$CUR_NP" -le 2 ] && [ "$N" -gt 0 ]; then
      F=$(free_gpus); NF_=$(echo "$F" | grep -c .)
      if [ "$NF_" -ge 4 ]; then
        sleep 300
        F2=$(free_gpus); C=$(comm -12 <(echo "$F" | sort) <(echo "$F2" | sort)); NC=$(echo "$C" | grep -c .)
        if [ "$NC" -ge 4 ]; then
          log "upgrade window: killing 2-card run at step $N, moving to 4 cards"
          pkill -u yuzhewu -f "bidirectional.trai[n]"; sleep 30
          pkill -9 -u yuzhewu -f "bidirectional.trai[n]" 2>/dev/null; sleep 10
          REMAIN=$((BUDGET - N)); [ "$REMAIN" -lt 100 ] && REMAIN=100
          launch_ca "$(echo "$C" | head -4 | paste -sd, -)" "runs/v2_ca_bidir/step-$N.safetensors" "$REMAIN"
          sleep 900
        fi
      fi
    fi
  else
    MISS=$((MISS+1))
    if [ "$MISS" -ge 2 ]; then
      REMAIN=$((BUDGET - N)); [ "$REMAIN" -lt 100 ] && REMAIN=100
      GPUS=$(wait_for_cards)
      if [ "$N" -gt 0 ]; then
        launch_ca "$GPUS" "runs/v2_ca_bidir/step-$N.safetensors" "$REMAIN"
      else
        launch_ca "$GPUS" "" "$BUDGET"   # cold start from raw Wan2.2 (zhiyang parity)
      fi
      MISS=0; sleep 900
    fi
  fi
  sleep 300
done
log "KEEPER DONE: v2_ca_bidir at $BUDGET"
