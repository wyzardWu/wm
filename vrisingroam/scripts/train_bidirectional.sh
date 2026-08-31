#!/bin/bash
# Scoped bidirectional training: start from pretrained Wan2.2-TI2V-5B, train
# the whole DiT (action embedders cold-start inside it).
# Usage: bash scripts/train_bidirectional.sh <processed-out-root> <run-name> [extra args...]
set -e
cd "$(dirname "$0")/.."
. env.sh
OUT_ROOT=${1:?usage: train_bidirectional.sh <processed-out-root> <run-name>}
RUN=${2:?run name}
shift 2

# NPROC=n to train on n GPUs (default 8); pin cards with CUDA_VISIBLE_DEVICES.
NPROC=${NPROC:-8}
MULTI=""
[ "$NPROC" -gt 1 ] && MULTI="--multi_gpu"
accelerate launch --num_processes "$NPROC" $MULTI --mixed_precision bf16 \
  -m ReactiveGWM_Code.training.bidirectional.train \
  --game vrising \
  --dataset_base_path "$OUT_ROOT" \
  --dataset_metadata_path "$OUT_ROOT/metadata.csv" \
  --model_paths "$DIT_JSON" \
  --tokenizer_path "$TOKENIZER" \
  --height 480 --width 832 --num_frames 101 \
  --action_hold_window 1 \
  --use_cached_dataset --cache_root "$OUT_ROOT/cache" \
  --output_path "runs/$RUN" \
  --trainable_models dit \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --num_epochs 1000 \
  --max_train_steps 30000 \
  --save_steps 1000 \
  --gradient_accumulation_steps 1 \
  --action_dropout_prob 0.1 \
  --use_gradient_checkpointing \
  --save_full_state \
  --dataset_num_workers 4 \
  "$@"
