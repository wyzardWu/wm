#!/bin/bash
# 8/7 night orchestrator (runs on node2 via nohup):
#   Phase A: keep zn2 (runs/cf_stage1_zn2, GPUs 1,2,6,7) alive to step-5000.
#   Verdict: default sanity rollout on step-5000, measure dy via phase corr.
#            PASS = dy >= +0.60 (GT for this clip: dy ≈ +1.1..+1.5).
#   PASS -> keep zn2 alive to 10000, then run 10k eval set (sanity-10s +
#           manor custom 10s), measure, leave results in eval_student/.
#   FAIL -> stop zn2, launch bidirectional CA retrain (v2_ca_bidir, table
#           mode, init from v1 34k teacher) on 1,2,6,7 with 30k budget and
#           keep IT alive (weights-only resume, remaining-budget pattern).
# All decisions and measurements are appended to runs/night_auto2.log.
cd "$(dirname "$0")/.."
. env.sh
PY=~/miniconda3/envs/rgwm/bin/python
E=/data/yuzhewu/vrisingroam/eval_student
R=runs/cf_stage1_zn2
YAML=ReactiveGWM/training/causal_forcing/configs/vrising_stage1_zn.yaml
TABLE=/data/yuzhewu/vrisingroam/processed/action_context_table_v1.pt
LOG=runs/night_auto2.log
log(){ echo "$(date '+%m-%d %H:%M') $*" >> "$LOG"; }
state_ok(){ d=$1; [ -d "$d" ] && [ -z "$(ls "$d"/.tmp* 2>/dev/null)" ] \
    && [ -f "$d/model.safetensors" ] && [ -f "$d/optimizer.bin" ]; }

relaunch_zn2(){
  LATEST=""
  for d in $(ls -d $R/state-* 2>/dev/null | sort -t- -k2 -n -r); do
    state_ok "$d" && { LATEST=$d; break; }
  done
  [ -z "$LATEST" ] && { log "zn2: no complete state yet, waiting"; return; }
  for i in 0 1 2 3; do
    [ -f "$LATEST/random_states_$i.pkl" ] || cp "$LATEST/random_states_0.pkl" "$LATEST/random_states_$i.pkl"
  done
  log "zn2 relaunch from $LATEST"
  BASE_CKPT=runs/v1_full_ext/step-4000.safetensors \
  WAN_BASE_DIR=/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B \
  TOKENIZER_DIR=/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl \
  DATA_ROOT=data/processed/combined_5d \
  METADATA_PATH=data/processed/combined_5d/metadata_train.csv \
  OUT=$R CUDA_VISIBLE_DEVICES=1,2,6,7 nohup accelerate launch \
    --num_processes 4 --multi_gpu --mixed_precision bf16 \
    -m ReactiveGWM_Code.training.causal_forcing.train --stage ar_tf \
    --config $YAML --resume_state "$LATEST" \
    >> runs/cf_stage1_zn2_launch.log 2>&1 < /dev/null &
}

zn2_alive(){ pgrep -u yuzhewu -f "causal_forcing.trai[n]" >/dev/null; }

keep_until(){  # $1 = target step file
  MISS=0
  while [ ! -f "$1" ]; do
    if zn2_alive; then MISS=0; else
      MISS=$((MISS+1))
      [ "$MISS" -ge 2 ] && { relaunch_zn2; MISS=0; sleep 600; }
    fi
    sleep 300
  done
}

sanity_and_measure(){  # $1 ckpt  $2 out.mp4  $3 logfile  $4 extra args  $5 gpu list "0 1"
  for GPU in $5; do
    CUDA_VISIBLE_DEVICES=$GPU $PY \
      ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/inference/sanity_sample.py \
      --ckpt "$1" --config $YAML --out "$2" --device gpu_shared \
      --memory_fraction 0.3 $4 >> "$3" 2>&1
    [ -f "$2" ] && break
    echo "[night] sanity on GPU$GPU failed, trying next" >> "$3"
  done
  $PY scripts/measure_motion.py "$2" 2>>"$3"
}

# Probe clips (picked 8/7 by scanning metadata_train for verified GT motion;
# default sanity clip row0 turned out to be a nearly-static blocked clip):
#   W-probe row5  : pure W(92) , GT (this tool) dx=+0.43  dy=+7.87
#   D-probe row64 : pure D(101), GT (this tool) dx=-12.04 dy=+0.01
ET=/data/yuzhewu/vrisingroam/eval_teacher
WP_IMG=$ET/probeW_row5.png
WP_ACT=data/processed/combined_5d/actions/chunk_005/20260731_213546_491_chunk_005_00541535.parquet
DP_IMG=$ET/probeD_row64.png
DP_ACT=data/processed/combined_5d/actions/chunk_005/20260731_213546_491_chunk_005_00559412.parquet

# ---------------- Phase A: to step-5000 ----------------
log "night_auto2 up; waiting for zn2 step-5000"
keep_until $R/step-5000.safetensors
log "step-5000 present; running verdict probes"

MW=$(sanity_and_measure $R/step-5000.safetensors $E/zn2_5k_probeW.mp4 $E/zn2_5k_probeW.log \
     "--first_frame_image $WP_IMG --actions_parquet $WP_ACT --latent_frames 26" "0 1")
log "W-probe (GT dy=+7.87): $MW"
MD=$(sanity_and_measure $R/step-5000.safetensors $E/zn2_5k_probeD.mp4 $E/zn2_5k_probeD.log \
     "--first_frame_image $DP_IMG --actions_parquet $DP_ACT --latent_frames 26" "0 1")
log "D-probe (GT dx=-12.04): $MD"
DY=$(echo "$MW" | grep -o 'dy=[+-][0-9.]*' | cut -c4-)
# Gate: W-probe dy >= +1.5 (GT +7.87; noise band ±0.5; generous to
# weak-but-alive following). D-probe logged for the morning decision only.
# Missing measurement (eval crashed on both GPUs) defaults to FAIL: the CA
# retrain is the strategically preferred branch and the night must not stall.
PASS=$($PY -c "print(1 if float('${DY:-0}') >= 1.5 else 0)" 2>/dev/null || echo 0)

if [ "$PASS" = "1" ]; then
  # ---------------- PASS branch ----------------
  log "VERDICT PASS (dy=$DY >= 1.5): zn2 continues to 10000"
  keep_until $R/step-10000.safetensors
  log "step-10000 done; running 10k eval (GPUs now free)"
  M1=$(sanity_and_measure $R/step-10000.safetensors $E/zn2_10k_probeW10s.mp4 \
       $E/zn2_10k_probeW.log \
       "--first_frame_image $WP_IMG --actions_parquet $WP_ACT --latent_frames 50" "1 0")
  log "10k W-probe 10s: $M1"
  M3=$(sanity_and_measure $R/step-10000.safetensors $E/zn2_10k_probeD10s.mp4 \
       $E/zn2_10k_probeD.log \
       "--first_frame_image $DP_IMG --actions_parquet $DP_ACT --latent_frames 50" "1 0")
  log "10k D-probe 10s: $M3"
  M2=$(CUDA_VISIBLE_DEVICES=1 $PY \
    ReactiveGWM/DiffSynth-Studio/examples/ReactiveGWM_casual_forcing/inference/sanity_sample.py \
    --ckpt $R/step-10000.safetensors --config $YAML \
    --out $E/zn2_10k_manor10s.mp4 --device gpu_shared --memory_fraction 0.5 \
    --first_frame_image /data/yuzhewu/vrisingroam/eval_teacher/manor_seed.png \
    --actions_parquet /data/yuzhewu/vrisingroam/eval_teacher/custom_W8D6W6.parquet \
    --latent_frames 50 > $E/zn2_10k_manor.log 2>&1; \
    $PY scripts/measure_motion.py $E/zn2_10k_manor10s.mp4 2>/dev/null)
  log "10k manor10s: $M2"
  log "NIGHT DONE (PASS path)"
else
  # ---------------- FAIL branch ----------------
  log "VERDICT FAIL (dy=$DY < 0.60): stopping zn2, launching v2_ca_bidir"
  pkill -u yuzhewu -f "causal_forcing.trai[n]"; sleep 60
  pkill -9 -u yuzhewu -f "causal_forcing.trai[n]" 2>/dev/null; sleep 30

  BUDGET=30000
  launch_ca(){  # $1 = resume ckpt or empty, $2 = remaining steps
    EXTRA=""
    [ -n "$1" ] && EXTRA="--resume_from_ckpt $1"
    log "launch v2_ca_bidir remain=$2 ckpt=${1:-v1-teacher}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=1,2,6,7 NPROC=4 nohup bash scripts/train_bidirectional.sh \
      data/processed/combined_5d v2_ca_bidir \
      --dataset_metadata_path data/processed/combined_5d/metadata_train.csv \
      --action_context_table $TABLE \
      $EXTRA --max_train_steps "$2" --action_dropout_prob 0.1 \
      >> runs/v2_ca_launch.log 2>&1 < /dev/null &
  }
  launch_ca runs/v1_full_ext/step-4000.safetensors $BUDGET

  # Keeper for v2_ca_bidir (weights-only resume + remaining budget).
  MISS=0
  while :; do
    N=$(ls runs/v2_ca_bidir/step-*.safetensors 2>/dev/null \
        | sed 's/.*step-//; s/\.safetensors//' | sort -n | tail -1); N=${N:-0}
    [ "$N" -ge "$BUDGET" ] && { log "v2_ca_bidir budget complete ($N)"; break; }
    if pgrep -u yuzhewu -f "bidirectional.trai[n]" >/dev/null; then MISS=0; else
      MISS=$((MISS+1))
      if [ "$MISS" -ge 2 ]; then
        REMAIN=$((BUDGET - N)); [ "$REMAIN" -lt 100 ] && REMAIN=100
        if [ "$N" -gt 0 ]; then
          launch_ca "runs/v2_ca_bidir/step-$N.safetensors" "$REMAIN"
        else
          launch_ca runs/v1_full_ext/step-4000.safetensors "$BUDGET"
        fi
        MISS=0; sleep 900
      fi
    fi
    sleep 300
  done
  log "NIGHT DONE (FAIL path): v2_ca_bidir at 30k, ready for eval"
fi
