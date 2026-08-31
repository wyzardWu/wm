#!/bin/bash
# 自动归档:runs/abot8k_adaln 每出一份 lora+state 就拷到 keep_ckpts/abot8k_adaln1(写保护)。
set -u
SRC=/data/yuzhewu/ltxwm/runs/abot8k_adaln/checkpoints
DST=/data/yuzhewu/ltxwm/keep_ckpts/abot8k_adaln1
mkdir -p $DST
while true; do
  for f in $SRC/lora_weights_step_*.safetensors $SRC/training_state_step_*.pt $SRC/action_adaln_step_*.pt; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    if [ ! -f "$DST/$b" ]; then
      sleep 20   # 等写完
      cp "$f" "$DST/$b.tmp" && mv "$DST/$b.tmp" "$DST/$b" && chmod a-w "$DST/$b"
      echo "[archive] $b $(date)"
    fi
  done
  sleep 120
done
