#!/bin/bash
# Single-GPU T5-only precompute for the SF3 `visual` column.
# Writes new T5 .pt files under <cache_dir>/t5/<hh>/ and a sidecar
# manifest_visual.json. Does NOT touch existing manifest.json,
# video/, or first_frame/.
#
# GPU 3 is the default — at the time of writing it has the lowest
# compute utilization (~30%) among GPUs occupied by other users'
# training. Override with GPU=<idx> ./run_cache_visual_t5.sh
#
# Log:  /tmp/sf3_visual_t5_cache.log
# PID:  /tmp/sf3_visual_t5_cache.pid
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

GPU="${GPU:-3}"

CSV="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s/metadata_wo_pure_v5cat_10k_visual.csv"
CACHE_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s_cache_v5cat_480x832"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

# T5 + VAE (VAE required by pipeline loader; we drop it in-script).
SLIM_MODEL_PATHS_JSON="[\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]"

LOG=/tmp/sf3_visual_t5_cache.log
echo "=== launching visual T5 cache on GPU ${GPU} -> ${LOG} ==="

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

CUDA_VISIBLE_DEVICES=${GPU} \
PYTHONUNBUFFERED=1 \
nohup python examples/ReactiveGWM/sf3/extras/precompute_visual_t5_cache.py \
  --csv_path "${CSV}" \
  --cache_dir "${CACHE_ROOT}" \
  --prompt_column visual \
  --manifest_filename manifest_visual.json \
  --model_paths "${SLIM_MODEL_PATHS_JSON}" \
  --tokenizer_path "${TOKENIZER}" \
  --skip_existing \
  > ${LOG} 2>&1 &

PID=$!
echo "${PID}" > /tmp/sf3_visual_t5_cache.pid
echo "PID=${PID} log=${LOG}"
echo ""
echo "Monitor:"
echo "  tail -f ${LOG}"
echo "  ls ${CACHE_ROOT}/t5 | wc -l   # expect 256 dirs (00..ff hash buckets)"
echo "  find ${CACHE_ROOT}/t5 -name '*.pt' | wc -l   # expect ~16273 after run"
echo "  ls -la ${CACHE_ROOT}/manifest_visual.json"
