#!/usr/bin/env bash

set -euo pipefail

# The original reactivegwm environment is no longer present on this host.
# V2 is validated against the canonical DiffSynth environment.
export PYTHON_BIN="${PYTHON_BIN:-/home/zeqingwang/anaconda3/envs/diffsynth/bin/python}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-/home/zeqingwang/anaconda3/envs/diffsynth/bin/accelerate}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

METADATA="${REBUTTAL_DIR}/generated/metadata_v2_strategy_only.csv"
CACHE_ROOT="${CACHE_BASE}/v2"
OUT="${OUT:-${OUTPUT_BASE}/v2_strategy_only}"
PORT="${PORT:-29712}"

launch_formal_training v2 "${METADATA}" "${CACHE_ROOT}" "${OUT}" "${PORT}" "$@"
