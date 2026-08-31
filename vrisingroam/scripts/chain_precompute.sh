#!/bin/bash
# Waits for run_all to finish cutting all 62 chunks (005-066), then launches
# the 8-GPU cache precompute. Safe to re-run (skip_existing everywhere).
cd "$(dirname "$0")/.."
OUT=${1:-data/processed/20260731}
while true; do
  n=$(ls "$OUT/.done" 2>/dev/null | wc -l)
  echo "$(date +%H:%M) chunks done: $n/62"
  [ "$n" -ge 62 ] && break
  # run_all died? (no python process and not finished)
  if ! pgrep -f "vrising_data.run_all" >/dev/null; then
    echo "$(date +%H:%M) WARNING: run_all not running at $n/62 — restarting"
    nohup python3 -m vrising_data.run_all --session 20260731_213546_491 \
      --session_dir data/raw/20260731_213546_491 --out_root "$OUT" \
      --start 5 --end 66 --keep_raw >> "$OUT/run_all.log" 2>&1 &
  fi
  sleep 600
done
echo "$(date +%H:%M) cutting complete; starting precompute"
bash scripts/precompute_cache.sh "$OUT"
echo "$(date +%H:%M) ALL_DATA_READY"
