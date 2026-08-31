#!/bin/bash
# Mode A + cache: Full DiT fine-tune with precomputed VAE/T5 cache. 8 GPU (0-7), bs=1/GPU, accum=1 -> eff bs=8.
# Pre-req: scripts/precompute_cache.py finished writing to ${CACHE_ROOT}.
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

DATA_ROOT="${DATA_ROOT:-/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s}"
CACHE_ROOT="${CACHE_ROOT:-/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s_cache_v2}"
METADATA="${METADATA:-${DATA_ROOT}/metadata.csv}"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
OUT="${OUT:-/home/zeqingwang/zeqingwang/models/train/sf2/Baseline/full_5s_480x608}"

# Optional: resume DiT weights from a previous run's safetensors. Optimizer
# state is NOT restored. Default: cold start (empty).
RESUME_CKPT="${RESUME_CKPT-}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"

if [ ! -f "${CACHE_ROOT}/manifest.json" ]; then
  echo "ERROR: cache manifest missing at ${CACHE_ROOT}/manifest.json"
  echo "Run scripts/precompute_cache.py first."
  exit 1
fi

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

echo "=== SF2_final / Mode A+cache: Full DiT fine-tune (cached) ==="
echo "Data:   ${DATA_ROOT}"
echo "CSV:    ${METADATA}"
echo "Cache:  ${CACHE_ROOT}"
echo "Output: ${OUT}"
if [ -n "${RESUME_CKPT}" ]; then
  if [ ! -f "${RESUME_CKPT}" ]; then
    echo "ERROR: RESUME_CKPT not found: ${RESUME_CKPT}"
    exit 1
  fi
  echo "Resume: ${RESUME_CKPT}"
else
  echo "Resume: (cold start)"
fi
echo "Start:  $(date)"

PYTHONUNBUFFERED=1 accelerate launch --num_processes=8 --multi_gpu --main_process_port 29605 \
  examples/ReactiveGWM/model_training/train.py \
  --game sf2 \
  --dataset_base_path "${DATA_ROOT}" \
  --dataset_metadata_path "${METADATA}" \
  --data_file_keys "video,action" \
  --height 480 --width 608 --num_frames 101 \
  --dataset_repeat "${DATASET_REPEAT}" \
  --use_cached_dataset --cache_root "${CACHE_ROOT}" \
  --model_paths "[[\"${BASE}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]" \
  --tokenizer_path "/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --learning_rate 5e-5 \
  --num_epochs 1000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 1 \
  --output_path "${OUT}" \
  --use_gradient_checkpointing \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --action_hold_window 10 \
  --use_csv_prompt true \
  --prompt_column prompt \
  --dataset_num_workers 4 \
  ${RESUME_CKPT:+--resume_from_ckpt "${RESUME_CKPT}"}

echo "Finished at $(date)"
