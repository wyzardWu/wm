#!/bin/bash
# OOM 恢复: 把某个 rank 的 shard 转移到另一张 GPU 重启 (--skip_existing 续跑)。
#
# 用法:
#   ./restart_cache_shard_v5.sh <rank> <new_gpu> [--with-manifest]
#   # rank 0 默认带 T5+manifest，其余 rank 默认 --skip_manifest_write
#
# 会先 kill 旧 PID（从 /tmp/sf2_cache_v5_pids.txt 读），再在新 GPU 上重启。
set -e

if [ $# -lt 2 ]; then
  echo "用法: $0 <rank> <new_gpu> [--with-manifest]"
  exit 1
fi

RANK=$1
NEW_GPU=$2
EXTRA_FLAGS=""
[ "$RANK" != "0" ] && EXTRA_FLAGS="--skip_manifest_write"
[ "$3" = "--with-manifest" ] && EXTRA_FLAGS=""

cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio
DATA_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s"
CACHE_ROOT="/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s_cache_v5"
METADATA="${DATA_ROOT}/metadata_v5.csv"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

MODEL_PATHS_JSON="[[\"${BASE}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

# Kill 旧 PID
OLD_PID=$(grep "^shard rank=${RANK} " /tmp/sf2_cache_v5_pids.txt 2>/dev/null | awk -F'PID=' '{print $2}' | tail -1)
if [ -n "$OLD_PID" ]; then
  echo "killing old PID ${OLD_PID} (rank ${RANK})"
  kill -9 ${OLD_PID} 2>/dev/null || true
  sleep 2
fi

LOG=/tmp/sf2_cache_v5_rank${RANK}.log
echo "=== restarting rank=${RANK} on GPU ${NEW_GPU} -> ${LOG} ==="
CUDA_VISIBLE_DEVICES=${NEW_GPU} \
PYTHONUNBUFFERED=1 \
nohup python examples/ReactiveGWM/scripts/precompute_cache.py \
  --game sf2 \
  --metadata "${METADATA}" \
  --dataset_base "${DATA_ROOT}" \
  --cache_root "${CACHE_ROOT}" \
  --height 480 --width 608 --num_frames 101 \
  --use_csv_prompt true --prompt_column prompt \
  --model_paths "${MODEL_PATHS_JSON}" \
  --tokenizer_path "${TOKENIZER}" \
  --shard_rank ${RANK} --shard_world_size 8 \
  --skip_existing \
  ${EXTRA_FLAGS} \
  >> ${LOG} 2>&1 &
NEW_PID=$!
echo "shard rank=${RANK} GPU=${NEW_GPU} PID=${NEW_PID}" | tee -a /tmp/sf2_cache_v5_pids.txt
echo "tail -f ${LOG} 监控"
