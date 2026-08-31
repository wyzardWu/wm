#!/bin/bash
# Launch 8 parallel cache precompute shards on GPU 0-7 for clips_5s_cache_v2.
#
# Rank 0 (GPU 0): T5 phase (encodes all unique prompts) + 1/8 of VAE + writes manifest.
# Rank 1-7 (GPU 1..7): only VAE for their 1/8 slice; --skip_manifest_write.
# All shards write to the same cache_root; atomic writes + --skip_existing make it safe.
#
# Logs: /tmp/sf2_cache_v2_rank{0..7}.log
# PIDs written to /tmp/sf2_cache_v2_pids.txt
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

DATA_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s"
CACHE_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s_cache_v2"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

MODEL_PATHS_JSON="[[\"${BASE}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]"

mkdir -p "${CACHE_ROOT}"
> /tmp/sf2_cache_v2_pids.txt

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

run_shard() {
  local GPU=$1
  local RANK=$2
  local EXTRA_FLAGS=$3
  local LOG=/tmp/sf2_cache_v2_rank${RANK}.log
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
    --shard_rank ${RANK} --shard_world_size 8 \
    --skip_existing \
    ${EXTRA_FLAGS} \
    > ${LOG} 2>&1 &
  local PID=$!
  echo "shard rank=${RANK} GPU=${GPU} PID=${PID}" | tee -a /tmp/sf2_cache_v2_pids.txt
}

# Rank 0 owns T5 phase + manifest. Start it FIRST so T5 encoding gets a head start.
run_shard 0 0 ""
sleep 5

# Rank 1-7: VAE only, no manifest write.
run_shard 1 1 "--skip_manifest_write"
run_shard 2 2 "--skip_manifest_write"
run_shard 3 3 "--skip_manifest_write"
run_shard 4 4 "--skip_manifest_write"
run_shard 5 5 "--skip_manifest_write"
run_shard 6 6 "--skip_manifest_write"
run_shard 7 7 "--skip_manifest_write"

echo ""
echo "=== 8 shards launched ==="
cat /tmp/sf2_cache_v2_pids.txt
echo ""
echo "Monitor:"
echo "  tail -f /tmp/sf2_cache_v2_rank{0..7}.log"
echo "  watch -n 30 'find ${CACHE_ROOT} -name \"*.pt\" | wc -l'"
echo ""
echo "Expected on completion: 11067 video + 11067 first_frame + ~5000 t5 = ~27000+ .pt files"
