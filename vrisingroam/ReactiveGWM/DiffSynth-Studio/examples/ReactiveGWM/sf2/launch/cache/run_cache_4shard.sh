#!/bin/bash
# Launch 4 parallel cache precompute shards on GPU 4-7.
#
# Rank 0 (GPU 4): T5 phase (all 5072+1 unique prompts) + 1/4 of VAE + writes manifest.
# Rank 1-3 (GPU 5/6/7): only VAE for their slice; --skip_manifest_write.
# All shards write to the same cache_root; atomic writes + --skip_existing make it safe.
#
# Logs: /tmp/sf2_cache_rank{0..3}.log
# PIDs written to /tmp/sf2_cache_pids.txt for easy tracking.
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

DATA_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s"
CACHE_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s_cache"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

MODEL_PATHS_JSON="[[\"${BASE}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]"

mkdir -p "${CACHE_ROOT}"
> /tmp/sf2_cache_pids.txt

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

run_shard() {
  local GPU=$1
  local RANK=$2
  local EXTRA_FLAGS=$3
  local LOG=/tmp/sf2_cache_rank${RANK}.log
  echo "=== launching shard rank=${RANK} on GPU ${GPU} -> ${LOG} ==="
  CUDA_VISIBLE_DEVICES=${GPU} \
  PYTHONUNBUFFERED=1 \
  nohup python examples/ReactiveGWM/scripts/precompute_cache.py \
  --game sf2 \
    --metadata "${DATA_ROOT}/metadata.csv" \
    --dataset_base "${DATA_ROOT}" \
    --cache_root "${CACHE_ROOT}" \
    --height 480 --width 608 --num_frames 101 \
    --use_csv_prompt true --prompt_column prompt \
    --model_paths "${MODEL_PATHS_JSON}" \
    --tokenizer_path "${TOKENIZER}" \
    --shard_rank ${RANK} --shard_world_size 4 \
    --skip_existing \
    ${EXTRA_FLAGS} \
    > ${LOG} 2>&1 &
  local PID=$!
  echo "shard rank=${RANK} GPU=${GPU} PID=${PID}" | tee -a /tmp/sf2_cache_pids.txt
}

# Rank 0 owns T5 phase + manifest. Start it FIRST and wait briefly so T5
# encoding gets a head start before VAE shards spawn.
run_shard 4 0 ""
sleep 5

# Rank 1-3: VAE only, do not write manifest.
run_shard 5 1 "--skip_manifest_write"
run_shard 6 2 "--skip_manifest_write"
run_shard 7 3 "--skip_manifest_write"

echo ""
echo "=== 4 shards launched ==="
cat /tmp/sf2_cache_pids.txt
echo ""
echo "Monitor:"
echo "  tail -f /tmp/sf2_cache_rank{0..3}.log"
echo "  watch -n 30 'find ${CACHE_ROOT} -name \"*.pt\" | wc -l'"
echo ""
echo "Expected on completion: 10820 video + 10820 first_frame + 5073 t5 = 26713 .pt files"
