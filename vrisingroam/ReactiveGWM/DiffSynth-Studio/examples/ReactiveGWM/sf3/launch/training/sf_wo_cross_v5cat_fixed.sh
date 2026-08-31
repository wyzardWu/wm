#!/bin/bash
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# SF_wo_cross: scoped full-DiT training EXCLUDING cross_attn, with fixed
# SF3 prompt ("SF3 Game.") + cached T5/VAE. Cold start.
#
# Goals:
#   * Isolate the action-conditioning pathway: cross_attn frozen means the
#     prompt context contributes a constant residual, so all behavior diffs
#     between clips come from the keyboard_action input (+ self_attn and
#     other DiT params, which DO update).
#   * Fixed prompt ("SF3 Game.") = same T5 embedding every step. The 0.1
#     prompt_dropout still runs (per CFG-style training contract), so the
#     model also sees the empty-prompt embedding 10% of the time.
#
# Cache contract:
#   * dst cache built by examples/ReactiveGWM/sf3/extras/rebuild_cache_fixed_prompt.py
#     from the v5cat cache (clips_5s_cache_v5cat_480x832/).
#   * manifest.config.use_csv_prompt MUST be False to match the
#     --use_csv_prompt flag NOT being set below.
#
# VAE/T5 unload check:
#   After launch, grep stdout for: "[cached] dropped pipe.vae and pipe.text_encoder"
#   Expected within first ~30s. If absent, kill + investigate cache_root.
#
# Resolution: 480×832, num_frames=101 (5s @ 20fps + 1).
#   Source SF3 video: 384x224 -> ImageCropAndResize cover-mode upscale
#   (BILINEAR, scale=2.167) -> 480×832, ~2.7px crop top/bottom.

CSV="/opt/dlami/nvme/zeqingwang/datasets/SF3/train/clips_5s/metadata_wo_pure_v5cat_10k.csv"
DATA_BASE="/opt/dlami/nvme/zeqingwang/datasets/SF3/train/clips_5s"
CACHE_ROOT="/home/zeqingwang/zeqingwang/DiffSynth-Studio/examples/ReactiveGWM/sf3/cache_v5cat_480x832_fixed"
OUT_DIR="/opt/dlami/nvme/zeqingwang/models/train/SF_wo_cross"
BASE="/opt/dlami/nvme/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER="/opt/dlami/nvme/zeqingwang/models/base_model/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"

mkdir -p "${OUT_DIR}"

echo "=== SF_wo_cross — scoped full-DiT (NO cross_attn), fixed prompt 'SF3 Game.' ==="
echo "CSV:          ${CSV}"
echo "DATA_BASE:    ${DATA_BASE}"
echo "CACHE_ROOT:   ${CACHE_ROOT}"
echo "OUT_DIR:      ${OUT_DIR}"
echo "BASE:         ${BASE}"
echo "TOKENIZER:    ${TOKENIZER}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES}"
echo "Start:        $(date)"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

PYTHONUNBUFFERED=1 accelerate launch --num_processes=8 --multi_gpu --main_process_port 29530 \
  examples/ReactiveGWM/model_training/train.py \
  --game sf3 \
  --dataset_base_path "${DATA_BASE}" \
  --dataset_metadata_path "${CSV}" \
  --use_cached_dataset \
  --cache_root "${CACHE_ROOT}" \
  --data_file_keys "video,action" \
  --height 480 \
  --width 832 \
  --num_frames 101 \
  --dataset_repeat 1 \
  --model_paths "[[\"${BASE}/diffusion_pytorch_model-00001-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00002-of-00003.safetensors\",\"${BASE}/diffusion_pytorch_model-00003-of-00003.safetensors\"],\"${BASE}/models_t5_umt5-xxl-enc-bf16.pth\",\"${BASE}/Wan2.2_VAE.pth\"]" \
  --tokenizer_path "${TOKENIZER}" \
  --learning_rate 5e-5 \
  --num_epochs 1000 \
  --save_steps 1000 \
  --output_path "${OUT_DIR}" \
  --use_gradient_checkpointing \
  --trainable_models "dit" \
  --trainable_filter_exclude "cross_attn" \
  --extra_inputs "input_image" \
  --action_hold_window 10 \
  --action_dropout_prob 0.1 \
  --prompt_dropout_prob 0.1 \
  --dataset_num_workers 4

echo ""
echo "Training finished at $(date)"
