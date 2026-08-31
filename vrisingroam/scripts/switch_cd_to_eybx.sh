#!/bin/bash
# Seamless CD -> EYBX switch on GPUs 2,3 (user-approved 2026-08-24):
#   1. stop cd_keeper FIRST (else it relaunches CD), then the CD run
#   2. cache precompute for overfit4 on the freed cards (6 shards)
#   3. launch EYBX overfit training
# Run inside tmux/nohup; takes ~1h before training starts.
set -e
cd "$(dirname "$0")/.."

echo "[switch] stopping cd_keeper"
pkill -f "cd_keeper" || true
sleep 2
echo "[switch] stopping CD (launcher + trainer)"
pkill -f "launch_stage2_252" || true
pkill -f "stage2_cd/cd_gm" || true
for i in $(seq 1 60); do
  pgrep -f "stage2_cd/cd_gm" >/dev/null || break
  sleep 5
done
sleep 10
echo "[switch] GPUs 2,3 after stop:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2,3

echo "[switch] cache precompute (overfit4)"
GAME=eybx GPUS_OVERRIDE="2 3" PER_GPU=3 bash scripts/precompute_cache.sh /data/yuzhewu/eybxroam/overfit4 \
  || { echo "precompute FAILED"; exit 1; }

echo "[switch] launching EYBX overfit"
GPUS=2,3 NPROC=2 bash scripts/train_eybx_overfit.sh eybx_overfit_v1
