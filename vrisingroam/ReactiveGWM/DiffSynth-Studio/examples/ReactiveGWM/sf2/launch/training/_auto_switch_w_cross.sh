#!/bin/bash
# Watcher: poll every 10 min for step-40000.safetensors. When stable,
# atomically switch from Phase 1 (no_cross_fixed_prompt) to Phase 2 (w_cross).
#
# Safety:
#   * Kills ONLY processes owned by zeqingwang AND whose cmdline contains
#     'examples/ReactiveGWM/model_training/train.py'. Other users (binhan,
  --game sf2 \
#     zhaohux, etc.) and other zeqingwang processes (sam2_seg, etc.) are
#     untouched.
#   * Stability check: target file must be > 1 GB and at least 30 s old
#     before triggering (avoids racing a partial save).
#   * To abort the watcher: kill its PID. That alone does NOT touch Phase 1.
#   * Once the switch fires, the watcher exits.
set -u

REPO=/home/zeqingwang/zeqingwang/Training_Wan
TARGET=/home/zeqingwang/zeqingwang/models/train/sf2/Baseline/no_cross_fixed_prompt/step-40000.safetensors
NEW_SCRIPT="${REPO}/examples/ReactiveGWM/sf2/model_training/launch/baseline_w_cross_resume40k.sh"
LOGDIR=/home/zeqingwang/zeqingwang/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOGDIR}/sf2_phase_switch_${TS}.log"
NEW_LOG="${LOGDIR}/sf2_phase2_train_${TS}.log"

mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "Watcher started; PID=$$"
log "Polling interval: 600 s"
log "Target file: $TARGET"
log "Phase 2 script: $NEW_SCRIPT"
log "Phase 2 stdout/stderr: $NEW_LOG"

while true; do
  if [ -f "$TARGET" ]; then
    SIZE=$(stat -c '%s' "$TARGET" 2>/dev/null || echo 0)
    NOW=$(date +%s)
    MTIME=$(stat -c '%Y' "$TARGET" 2>/dev/null || echo 0)
    AGE=$(( NOW - MTIME ))
    if [ "$SIZE" -gt 1000000000 ] && [ "$AGE" -gt 30 ]; then
      log "Target stable (size=${SIZE}B age=${AGE}s). Triggering switch."
      break
    else
      log "Target seen but unstable (size=${SIZE}B age=${AGE}s). Waiting another cycle."
    fi
  fi
  sleep 600
done

log "[STEP 1/4] Identifying Phase 1 SF2 training PIDs (zeqingwang + train.py)"
PIDS=$(pgrep -u zeqingwang -f 'examples/ReactiveGWM/model_training/train.py' || true)
  --game sf2 \
log "PIDs to terminate: ${PIDS:-<none>}"

if [ -n "${PIDS}" ]; then
  log "[STEP 2/4] SIGTERM, then 30 s grace"
  echo "$PIDS" | xargs -r kill -TERM
  sleep 30
  REMAINING=$(pgrep -u zeqingwang -f 'examples/ReactiveGWM/model_training/train.py' || true)
  --game sf2 \
  if [ -n "${REMAINING}" ]; then
    log "  still alive: ${REMAINING}; sending SIGKILL"
    echo "$REMAINING" | xargs -r kill -KILL
    sleep 10
  else
    log "  all Phase 1 procs exited cleanly"
  fi
else
  log "  (no matching procs; Phase 1 already stopped)"
fi

log "[STEP 3/4] GPU snapshot:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader >> "$LOG" 2>&1 || true

log "[STEP 4/4] Launching Phase 2 (output -> $NEW_LOG)"
cd "$REPO"
nohup bash "$NEW_SCRIPT" > "$NEW_LOG" 2>&1 &
NEW_PID=$!
log "Phase 2 launched; PID=${NEW_PID}"
log "Watcher exiting."
