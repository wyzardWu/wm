#!/bin/bash
# 252: hold each GPU as it frees; once 4 are held, swap them to the
# from-scratch blocked-filtered teacher training. Retries until training sticks.
PY=$HOME/miniconda3/envs/rgwm/bin/python
SP=$HOME/vrisingroam/scripts
RUN=/data/yuzhewu/vrisingroam/teacher252/runs
LOGMAIN=$RUN/grab4.log
declare -A HPID
attempt=0
mkdir -p "$RUN"

while true; do
  for g in 0 1 2 3 4 5 6 7; do
    if [ -n "${HPID[$g]}" ] && kill -0 "${HPID[$g]}" 2>/dev/null; then continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$used" -lt 20000 ]; then
      "$PY" "$SP/hold_252.py" "$g" > /dev/null 2>&1 &
      HPID[$g]=$!
      echo "$(date +%H:%M) holding gpu$g pid ${HPID[$g]}" >> "$LOGMAIN"
    fi
  done
  HELD=()
  for g in 0 1 2 3 4 5 6 7; do
    [ -n "${HPID[$g]}" ] && kill -0 "${HPID[$g]}" 2>/dev/null && HELD+=("$g")
  done
  if [ "${#HELD[@]}" -ge 4 ]; then
    PICK=("${HELD[@]:0:4}")
    G4=$(IFS=,; echo "${PICK[*]}")
    attempt=$((attempt+1))
    LOG=$RUN/v3_noblock_scratch_try$attempt.log
    echo "$(date +%H:%M) attempt $attempt on GPUs $G4" >> "$LOGMAIN"
    for g in "${PICK[@]}"; do kill -9 "${HPID[$g]}" 2>/dev/null; unset "HPID[$g]"; done
    sleep 3
    cd "$HOME/vrisingroam"
    START=$(date +%s)
    CUDA_VISIBLE_DEVICES=$G4 NPROC=4 bash scripts/train_bidirectional.sh \
      /data/yuzhewu/vrisingroam/teacher252 v3_noblock_scratch \
      --dataset_metadata_path /data/yuzhewu/vrisingroam/teacher252/metadata_train_noblock02.csv \
      --action_context_table /data/yuzhewu/vrisingroam/processed/action_context_table_v3.pt \
      --output_path /data/yuzhewu/vrisingroam/teacher252/runs/v3_noblock_scratch \
      --max_train_steps 999000 > "$LOG" 2>&1 &
    LPID=$!
    while kill -0 "$LPID" 2>/dev/null; do
      AGE=$(( $(date +%s) - START ))
      if grep -q "it/s" "$LOG" || [ "$AGE" -gt 1200 ]; then
        echo "$(date +%H:%M) TRAINING UP attempt $attempt gpus $G4" >> "$LOGMAIN"
        pkill -f "hold_25[2]"
        exit 0
      fi
      if grep -qE "OutOfMemoryError|Traceback" "$LOG"; then
        pkill -9 -f "bidirectional.trai[n]"
        kill -9 "$LPID" 2>/dev/null
        echo "$(date +%H:%M) attempt $attempt failed, re-holding" >> "$LOGMAIN"
        sleep 20
        break
      fi
      sleep 30
    done
    if grep -q "it/s" "$LOG"; then
      pkill -f "hold_25[2]"
      exit 0
    fi
  fi
  sleep 20
done
