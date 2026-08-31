#!/bin/bash
# Launch 8 parallel SF3 v5cat cache precompute shards on GPU 0-7.
#
# Differences vs run_cache_8shard.sh:
#  * MODEL_PATHS_JSON contains ONLY T5 + VAE — NO DiT shards.
#    precompute drops DiT immediately anyway; this avoids the v5 OOM
#    we hit on GPU 0 / GPU 2 when other users had filled VRAM.
#  * cache_dir -> clips_5s_cache_v5cat_480x832 (mirror v5 naming)
#  * CSV       -> metadata_wo_pure_v5cat_10k.csv (10k post-balance)
#
# Rank 0 (GPU 0): T5 phase + 1/8 of VAE + writes manifest.
# Rank 1-7 (GPU 1-7): only VAE for their slice; --skip_manifest_write.
# All shards write same cache_root; atomic writes + --skip_existing safe.
# Does NOT touch other users' processes — only nohup-launches own 8 shards.
#
# Logs:  /tmp/sf3_v5cat_cache_rank{0..7}.log
# PIDs:  /tmp/sf3_v5cat_cache_pids.txt
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

CSV="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s/metadata_wo_pure_v5cat_10k.csv"
DATA_BASE="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s"
CACHE_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s_cache_v5cat_480x832"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

# Slim: T5 + VAE only. NO DiT.
SLIM_MODEL_PATHS_JSON="[\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]"

mkdir -p "${CACHE_ROOT}"
> /tmp/sf3_v5cat_cache_pids.txt

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

run_shard() {
  local GPU=$1
  local RANK=$2
  local EXTRA_FLAGS=$3
  local LOG=/tmp/sf3_v5cat_cache_rank${RANK}.log
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
    --model_paths "${SLIM_MODEL_PATHS_JSON}" \
    --tokenizer_path "${TOKENIZER}" \
    --shard_rank ${RANK} --shard_world_size 8 \
    --skip_existing \
    ${EXTRA_FLAGS} \
    > ${LOG} 2>&1 &
  local PID=$!
  echo "shard rank=${RANK} GPU=${GPU} PID=${PID}" | tee -a /tmp/sf3_v5cat_cache_pids.txt
}

# Rank 0 owns T5 phase + manifest. Start FIRST and head-start 5s.
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
cat /tmp/sf3_v5cat_cache_pids.txt
echo ""
echo "Monitor:"
echo "  tail -f /tmp/sf3_v5cat_cache_rank{0..7}.log"
echo "  watch -n 30 'find ${CACHE_ROOT} -name \"*.pt\" | wc -l'"
echo ""
echo "Expected on completion: 10000 video + 10000 first_frame + ~7300 t5 ≈ 27300 .pt files"
echo ""
echo "After ALL shards finish, run PSNR verify (single GPU, ~5 min):"
echo "  CUDA_VISIBLE_DEVICES=0 python examples/ReactiveGWM/scripts/precompute_cache.py \\"
  --game sf3 \
echo "      --csv_path \"${CSV}\" --dataset_base \"${DATA_BASE}\" --cache_dir \"${CACHE_ROOT}\" \\"
echo "      --height 480 --width 832 --num_frames 101 --use_csv_prompt --prompt_column prompt \\"
echo "      --model_paths '${SLIM_MODEL_PATHS_JSON}' --tokenizer_path \"${TOKENIZER}\" \\"
echo "      --skip_existing --verify_first 3 --skip_manifest_write"
