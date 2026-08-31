#!/bin/bash
# Baseline Phase 2: resume from step-40000 of `no_cross_fixed_prompt` and
# train ONLY the cross_attn (text->vision) blocks. Everything else frozen.
#
# Stage chain:
#   Phase 1 (running):  trainable_filter_exclude=cross_attn + fixed prompt
#                       -> burns in self_attn / ffn / action / etc.
#   Phase 2 (this run): trainable_filter=cross_attn        + CSV prompt
#                       -> turns on text path with varying prompts.
#
# Differences vs baseline_no_cross_fixed_prompt.sh:
#   * --resume_from_ckpt <step-40000.safetensors>          (warm start)
#   * --trainable_filter "cross_attn"                      (train cross_attn ONLY)
#   *   removed --trainable_filter_exclude
#   * CACHE_ROOT -> clips_5s_cache_v5                      (CSV-prompt cache)
#   * --use_csv_prompt true                                (varying prompts)
#   * OUT       -> .../Baseline/SF2_ALL/w_cross
#   * main_process_port -> 29609                           (avoid clash with 29608)
#
# Smoke checks (look for these in the first ~30s of stdout):
#   [Cold-start] Transferred N/M keys                      (still runs first)
#   [Resume] loaded K keys from .../step-40000.safetensors (...)
#   [Trainable Filter] keep=['cross_attn'] drop=[] -> X tensors trainable (Y scalars)
#   [cached] filtered 4 pipeline units
#   [cached] dropped pipe.vae and pipe.text_encoder

set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

DATA_ROOT="${DATA_ROOT:-/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s}"
CACHE_ROOT="${CACHE_ROOT:-/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s_cache_v5}"
METADATA="${METADATA:-${DATA_ROOT}/metadata_v5.csv}"
BASE="/home/zeqingwang/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
RESUME_CKPT="${RESUME_CKPT:-/home/zeqingwang/zeqingwang/models/train/sf2/Baseline/no_cross_fixed_prompt/step-40000.safetensors}"
OUT="${OUT:-/home/zeqingwang/zeqingwang/models/Baseline/SF2_ALL/w_cross}"

DATASET_REPEAT="${DATASET_REPEAT:-1}"

if [ ! -f "${CACHE_ROOT}/manifest.json" ]; then
  echo "ERROR: cache manifest missing at ${CACHE_ROOT}/manifest.json"
  exit 1
fi

if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: resume ckpt missing at ${RESUME_CKPT}"
  echo "       Wait for Phase 1 to reach step-40000 or override RESUME_CKPT=..."
  exit 1
fi

mkdir -p "${OUT}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

echo "=== SF2_final / Baseline Phase 2: resume @ step-40000, train ONLY cross_attn ==="
echo "Data:    ${DATA_ROOT}"
echo "CSV:     ${METADATA}"
echo "Cache:   ${CACHE_ROOT}              (use_csv_prompt=true)"
echo "Resume:  ${RESUME_CKPT}"
echo "Output:  ${OUT}"
echo "GPUs:    ${CUDA_VISIBLE_DEVICES}"
echo "Start:   $(date)"

PYTHONUNBUFFERED=1 accelerate launch --num_processes=8 --multi_gpu --main_process_port 29609 \
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
  --resume_from_ckpt "${RESUME_CKPT}" \
  --learning_rate 5e-5 \
  --num_epochs 1000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 1 \
  --output_path "${OUT}" \
  --use_gradient_checkpointing \
  --trainable_models "dit" \
  --trainable_filter "cross_attn" \
  --extra_inputs "input_image" \
  --action_hold_window 10 \
  --use_csv_prompt true \
  --prompt_column prompt \
  --dataset_num_workers 4 \
  --prompt_dropout_prob 0.1

echo "Finished at $(date)"
