#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

METADATA="${REBUTTAL_DIR}/generated/metadata_v1_vanilla_strategy.csv"
CACHE_ROOT="${CACHE_BASE}/v1"
OUT="${OUT:-${OUTPUT_BASE}/v1_vanilla_strategy}"
PORT="${PORT:-29711}"

for _ in $(seq 1 240); do
  if ! pgrep -f '[e]xamples/Rebuttal/supplemental_vae_worker.py' >/dev/null; then
    break
  fi
  echo "Waiting for supplemental V1 cache workers to exit before training..."
  sleep 5
done
if pgrep -f '[e]xamples/Rebuttal/supplemental_vae_worker.py' >/dev/null; then
  echo "ERROR: supplemental V1 cache workers did not exit within 20 minutes" >&2
  exit 1
fi

launch_formal_training v1 "${METADATA}" "${CACHE_ROOT}" "${OUT}" "${PORT}" "$@"
