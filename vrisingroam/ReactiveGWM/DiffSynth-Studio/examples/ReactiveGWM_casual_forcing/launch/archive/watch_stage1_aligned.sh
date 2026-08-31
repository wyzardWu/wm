#!/usr/bin/env bash
# 后台 watcher — 盯修正版 Stage 1 (tmux cf_stage1_aligned) 训练健康.
# 每 INTERVAL 秒检查: tmux 存活 / step 推进(停滞) / NaN-inf loss / 日志错误 /
#   /nfs 磁盘 / 训练完成. 正常写 HEARTBEAT; 异常/完成/退出写 ">>>ALERT<<<" 前缀行.
# Monitor 只 grep ">>>ALERT<<<" → 低噪音 + 覆盖崩溃/完成/watcher 自身退出(不静默).
set -uo pipefail

LOG="${LOG:-/nfs/zeqingwang/models/train/sf3_casual_forcing_2/stage1_ar/train_20260525_040611.log}"
SESS="${SESS:-cf_stage1_aligned}"
DISK_PATH="${DISK_PATH:-/nfs/zeqingwang/models/train}"
INTERVAL="${INTERVAL:-600}"      # 10 min
DISK_MIN_GB="${DISK_MIN_GB:-1500}"
STALL_SEC="${STALL_SEC:-1800}"   # step 30 min 没涨 = 停滞 (save_state 阻塞 ~50s 不会误报)
MAX_STEPS="${MAX_STEPS:-100000}"

emit(){ echo "[$(date '+%F %T')] $*"; }
alert(){ echo "[$(date '+%F %T')] >>>ALERT<<< $*"; }
trap 'alert "watcher 退出 (pid=$$) — 监控已停止"' EXIT
trap 'exit 0' INT TERM

last_step=-1
last_change=$(date +%s)
emit "WATCHER START pid=$$ interval=${INTERVAL}s sess=$SESS"
emit "watch: $LOG"

while true; do
  now=$(date +%s)

  # 1. tmux 训练 session 存活
  tmux has-session -t "$SESS" 2>/dev/null || alert "tmux '$SESS' GONE — 训练 session 消失 (结束/崩溃?)"

  # 2. 最新 step + loss
  line=$(grep -oE "\[step [0-9]+\] loss=[-0-9.eEnafi]+" "$LOG" 2>/dev/null | tail -1)
  step=$(printf '%s' "$line" | grep -oE "step [0-9]+" | grep -oE "[0-9]+")
  loss=$(printf '%s' "$line" | sed -nE 's/.*loss=([-0-9.eEnafi]+).*/\1/p')

  # 3. NaN/inf 发散
  printf '%s' "$loss" | grep -qiE "nan|inf" && alert "loss=$loss @ step=$step — NaN/inf 发散!"

  # 4. 训练完成
  if grep -q "\[Done\] reached step=" "$LOG" 2>/dev/null; then
    alert "TRAINING DONE — 日志出现 [Done]; 末 step=${step:-?}. watcher 退出."; exit 0
  fi
  if [ -n "${step:-}" ] && [ "$step" -ge "$MAX_STEPS" ] 2>/dev/null; then
    alert "TRAINING 到 MAX_STEPS=$MAX_STEPS (step=$step). watcher 退出."; exit 0
  fi

  # 5. step 停滞 (hang/死锁)
  if [ -n "${step:-}" ]; then
    if [ "$step" = "$last_step" ]; then
      [ $((now - last_change)) -gt "$STALL_SEC" ] && { alert "step 停滞在 $step 超过 $((STALL_SEC/60)) 分钟 — 可能 hang"; last_change=$now; }
    else
      last_step=$step; last_change=$now
    fi
  fi

  # 6. 日志末尾错误
  errline=$(tail -120 "$LOG" 2>/dev/null | grep -iE "Traceback|CUDA error|out of memory|RuntimeError|NCCL|AssertionError|got signal|Killed" | tail -1)
  [ -n "$errline" ] && alert "日志错误: $errline"

  # 7. 磁盘 (df -P 强制单行, -BG 以 GB 计)
  avail=$(df -P -BG "$DISK_PATH" 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4); print $4}')
  [ -n "${avail:-}" ] && [ "$avail" -lt "$DISK_MIN_GB" ] 2>/dev/null && alert "/nfs 剩 ${avail}GB < ${DISK_MIN_GB}GB — 盘快满(ckpt 累积 ~2.5TB)"

  # 8. GPU 0-3 简报
  gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -4 | awk -F', ' '{printf "%s:%s/%s ",$1,$2,$3}')

  emit "HEARTBEAT step=${step:-?} loss=${loss:-?} disk=${avail:-?}GB gpu=[${gpu:-?}]"
  sleep "$INTERVAL"
done
