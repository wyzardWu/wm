#!/bin/bash
# Runs cache precompute incrementally on free GPUs while run_all is still
# cutting, so most of the cache exists by the time cutting finishes. The
# chain watcher's final precompute pass (flock-serialized) fills the tail
# and writes the manifest.
cd "$(dirname "$0")/.."
OUT=${1:-data/processed/20260731}
while true; do
  n=$(ls "$OUT/.done" 2>/dev/null | wc -l)
  [ "$n" -ge ${N_CHUNKS:-62} ] && { echo "$(date +%H:%M) cutting finished; leaving final pass to chain"; break; }
  echo "$(date +%H:%M) filler pass at $n/62 chunks"
  bash scripts/precompute_cache.sh "$OUT"
  sleep 120
done
