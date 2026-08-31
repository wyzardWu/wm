#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

METADATA="${DATA_ROOT}/metadata.csv"
CACHE_ROOT="${CACHE_BASE}/v3"
OUT="${OUT:-${OUTPUT_BASE}/v3_hybrid_cross_lora}"
PORT="${PORT:-29713}"

launch_formal_training v3 "${METADATA}" "${CACHE_ROOT}" "${OUT}" "${PORT}" "$@"
