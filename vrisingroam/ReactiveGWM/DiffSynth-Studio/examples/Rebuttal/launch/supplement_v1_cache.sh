#!/usr/bin/env bash

set -euo pipefail
REBUTTAL_LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REBUTTAL_LAUNCH_DIR}/_common.sh"

V1_METADATA="${REBUTTAL_DIR}/generated/metadata_v1_vanilla_strategy.csv"
V1_CACHE_ROOT="${CACHE_BASE}/v1"
V1_MANIFEST="${V1_CACHE_ROOT}/manifest.json"
SUPPLEMENTAL_LAYOUT="${SUPPLEMENTAL_LAYOUT:-gpu0-a:0,gpu0-b:0,gpu0-c:0,gpu1-a:1}"
SUPPLEMENTAL_CPU_THREADS="${SUPPLEMENTAL_CPU_THREADS:-6}"
SUPPLEMENTAL_MIN_CSV_INDEX="${SUPPLEMENTAL_MIN_CSV_INDEX:-0}"
VAE_ONLY_MODEL_PATHS="[\"${VAE_PATH}\"]"

require_file "${V1_METADATA}"
require_file "${VAE_PATH}"
require_directory "${V1_CACHE_ROOT}"
if [[ -f "${V1_MANIFEST}" ]]; then
  echo "V1 cache manifest already exists; supplemental workers are unnecessary."
  exit 0
fi

IFS=',' read -r -a SUPPLEMENTAL_SPECS <<< "${SUPPLEMENTAL_LAYOUT}"
if [[ "${#SUPPLEMENTAL_SPECS[@]}" -eq 0 ]]; then
  echo "ERROR: SUPPLEMENTAL_LAYOUT must select at least one worker" >&2
  exit 1
fi

cd "${REPO_ROOT}"
worker_pids=()
terminate_workers() {
  local pid
  for pid in "${worker_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap terminate_workers INT TERM

for spec in "${SUPPLEMENTAL_SPECS[@]}"; do
  IFS=':' read -r worker_id gpu <<< "${spec}"
  if [[ -z "${worker_id}" || -z "${gpu}" ]]; then
    echo "ERROR: malformed supplemental worker spec: ${spec}" >&2
    terminate_workers
    exit 1
  fi
  log="${V1_CACHE_ROOT}/supplemental-${worker_id}.log"
  echo "Launching supplemental worker=${worker_id} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS="${SUPPLEMENTAL_CPU_THREADS}" \
    MKL_NUM_THREADS="${SUPPLEMENTAL_CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${SUPPLEMENTAL_CPU_THREADS}" \
    NUMEXPR_NUM_THREADS="${SUPPLEMENTAL_CPU_THREADS}" \
    PYTHONUNBUFFERED=1 \
    nice -n 10 "${PYTHON_BIN}" examples/Rebuttal/supplemental_vae_worker.py \
      --metadata "${V1_METADATA}" \
      --dataset_base "${DATA_ROOT}" \
      --cache_root "${V1_CACHE_ROOT}" \
      --vae_model_paths "${VAE_ONLY_MODEL_PATHS}" \
      --tokenizer_path "${TOKENIZER_PATH}" \
      --worker_id "${worker_id}" \
      --height 480 \
      --width 608 \
      --num_frames 101 \
      --cpu_threads "${SUPPLEMENTAL_CPU_THREADS}" \
      --min_csv_index "${SUPPLEMENTAL_MIN_CSV_INDEX}" \
      >"${log}" 2>&1 &
  worker_pids+=("$!")
done

echo "Supplemental worker PIDs: ${worker_pids[*]}"
failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
trap - INT TERM

if [[ "${failed}" -ne 0 ]]; then
  echo "ERROR: one or more supplemental V1 workers failed" >&2
  exit 1
fi
echo "All supplemental V1 workers exited cleanly."
