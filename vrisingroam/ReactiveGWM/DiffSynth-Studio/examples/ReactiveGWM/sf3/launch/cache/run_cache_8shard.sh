#!/bin/bash
# Launch 8 parallel SF3 v5 cache precompute shards on GPU 0-7.
#
# Rank 0 (GPU 0): T5 phase (all unique prompts) + 1/8 of VAE + writes manifest.
# Rank 1-7 (GPU 1-7): only VAE for their slice; --skip_manifest_write.
# All shards write to the same cache_root; atomic writes + --skip_existing make it safe.
#
# This launcher does NOT touch other users' processes — only nohup-launches its
# own 8 shards in the background. Use kill on PIDs in /tmp/sf3_v5_cache_pids.txt
# if you ever need to stop them; nothing else is killed.
#
# Verify (PSNR sanity) is run SEPARATELY after all shards finish — see the
# helper command printed at the end.
#
# Logs:  /tmp/sf3_v5_cache_rank{0..7}.log
# PIDs:  /tmp/sf3_v5_cache_pids.txt
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

CSV="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s/metadata_wo_pure_10k_v5.csv"
DATA_BASE="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s"
CACHE_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s_cache_v5_480x832"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

MODEL_PATHS_JSON="[[\"${BASE}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]"

mkdir -p "${CACHE_ROOT}"
> /tmp/sf3_v5_cache_pids.txt

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

run_shard() {
  local GPU=$1
  local RANK=$2
  local EXTRA_FLAGS=$3
  local LOG=/tmp/sf3_v5_cache_rank${RANK}.log
  echo "=== launching shard rank=${RANK} on GPU ${GPU} -> ${LOG} ==="
  CUDA_VISIBLE_DEVICES=${GPU} \
  PYTHONUNBUFFERED=1 \
  nohup python examples/ReactiveGWM/scripts/precompute_cache.py \
  --game sf3 \
    --csv_path "${CSV}" \
    --dataset_base "${DATA_BASE}" \
    --cache_dir "${CACHE_ROOT}" \
    --height 480 --width 832 --num_frames 101 \
    --height_division_factor 16 --width_division_factor 16 \
    --use_csv_prompt --prompt_column prompt \
    --model_paths "${MODEL_PATHS_JSON}" \
    --tokenizer_path "${TOKENIZER}" \
    --shard_rank ${RANK} --shard_world_size 8 \
    --skip_existing \
    ${EXTRA_FLAGS} \
    > ${LOG} 2>&1 &
  local PID=$!
  echo "shard rank=${RANK} GPU=${GPU} PID=${PID}" | tee -a /tmp/sf3_v5_cache_pids.txt
}

# Rank 0 owns T5 phase + manifest. Start FIRST and give it a 5s head-start
# so T5 encoding begins before sibling VAE shards spin up.
run_shard 0 0 ""
sleep 5

# Rank 1-7: VAE only, do not write manifest.
run_shard 1 1 "--skip_manifest_write"
run_shard 2 2 "--skip_manifest_write"
run_shard 3 3 "--skip_manifest_write"
run_shard 4 4 "--skip_manifest_write"
run_shard 5 5 "--skip_manifest_write"
run_shard 6 6 "--skip_manifest_write"
run_shard 7 7 "--skip_manifest_write"

echo ""
echo "=== 8 shards launched ==="
cat /tmp/sf3_v5_cache_pids.txt
echo ""
echo "Monitor:"
echo "  tail -f /tmp/sf3_v5_cache_rank{0..7}.log"
echo "  watch -n 30 'find ${CACHE_ROOT} -name \"*.pt\" | wc -l'"
echo ""
echo "Expected on completion: 10002 video + 10002 first_frame + 7320 t5 = 27324 .pt files"
echo ""
echo "After ALL shards finish, run PSNR verify (single GPU, ~5 min):"
echo "  CUDA_VISIBLE_DEVICES=0 python examples/ReactiveGWM/scripts/precompute_cache.py \\"
  --game sf3 \
echo "      --csv_path \"${CSV}\" --dataset_base \"${DATA_BASE}\" --cache_dir \"${CACHE_ROOT}\" \\"
echo "      --height 480 --width 832 --num_frames 101 \\"
echo "      --use_csv_prompt --prompt_column prompt \\"
echo "      --model_paths \"\${MODEL_PATHS_JSON}\" --tokenizer_path \"${TOKENIZER}\" \\"
echo "      --skip_existing --verify_first 3 --skip_manifest_write"
