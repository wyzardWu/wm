#!/bin/bash
# Prune eybx_overfit_v3 checkpoints while the uncapped run continues:
#   state-*   keep the 2 newest (resume safety)
#   step-*.safetensors  keep multiples of 2000 micro (probe grid) + 2 newest
RUN=~/vrisingroam/runs/eybx_overfit_v3
while true; do
  states=$(ls -dt "$RUN"/state-* 2>/dev/null)
  echo "$states" | tail -n +3 | xargs -r rm -rf
  weights=$(ls -t "$RUN"/step-*.safetensors 2>/dev/null)
  keepers=$(echo "$weights" | head -2)
  for w in $weights; do
    n=$(basename "$w" | sed 's/step-\([0-9]*\).safetensors/\1/')
    if [ $((n % 2000)) -ne 0 ] && ! echo "$keepers" | grep -q "$w"; then
      rm -f "$w"
    fi
  done
  sleep 300
done
