#!/bin/bash
# Compensates for a run_all instance launched with --keep_raw: deletes each
# raw chunk as soon as its .done marker appears, keeping disk usage flat.
# Exits once all 62 chunks (005-066) are done and cleaned.
cd "$(dirname "$0")/.."
RAW=data/raw/20260731_213546_491/video
DONE=data/processed/20260731/.done
while true; do
  for marker in "$DONE"/chunk_*.mp4.done; do
    [ -e "$marker" ] || continue
    raw="$RAW/$(basename "$marker" .done)"
    if [ -e "$raw" ]; then
      rm -f "$raw"
      echo "$(date +%H:%M) removed raw $(basename "$raw")"
    fi
  done
  n=$(ls "$DONE" 2>/dev/null | wc -l)
  if [ "$n" -ge 62 ] && ! ls "$RAW"/chunk_*.mp4 >/dev/null 2>&1; then
    echo "$(date +%H:%M) all chunks done and cleaned; exiting"
    break
  fi
  sleep 120
done
