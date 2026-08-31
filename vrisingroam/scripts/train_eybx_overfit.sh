#!/bin/bash
# EYBX scene-switch overfit: Wan2.2-TI2V-5B + per-frame CA context =
# [action 16 tok] ⊕ [scene 16 tok]. Data: overfit4 (900 normal + 300
# screen-verified splices, 3 scenes). Actions are SCREEN-space relabeled.
# Usage: [GPUS=2,3] [NPROC=2] bash scripts/train_eybx_overfit.sh [run-name] [extra args...]
set -e
cd "$(dirname "$0")/.."
. env.sh
RUN=${1:-eybx_overfit_v1}
[ $# -ge 1 ] && shift

OUT_ROOT=/data/yuzhewu/eybxroam/overfit4
TABLES=/data/yuzhewu/eybxroam/tables
GPUS="${GPUS:-2,3}" NPROC="${NPROC:-2}" PORT="${PORT:-29591}"
MULTI=""
[ "$NPROC" -gt 1 ] && MULTI="--multi_gpu"

CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes "$NPROC" $MULTI --main_process_port "$PORT" \
  --mixed_precision bf16 \
  -m ReactiveGWM_Code.training.bidirectional.train \
  --game eybx \
  --dataset_base_path "$OUT_ROOT" \
  --dataset_metadata_path "$OUT_ROOT/metadata.csv" \
  --model_paths "$DIT_JSON" \
  --tokenizer_path "$TOKENIZER" \
  --height 480 --width 832 --num_frames 101 \
  --action_hold_window 1 \
  --use_cached_dataset --cache_root "$OUT_ROOT/cache" \
  --action_context_table "$TABLES/eybx_action_table.pt" \
  --scene_context_table "$TABLES/eybx_scene_table.pt" \
  --action_dropout_prob 0.1 \
  --scene_dropout_prob 0.1 \
  --output_path "runs/$RUN" \
  --trainable_models dit \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --num_epochs 1000 \
  --max_train_steps 6000 \
  --save_steps 500 \
  --gradient_accumulation_steps 2 \
  --use_gradient_checkpointing \
  --save_full_state \
  --dataset_num_workers 4 \
  "$@"
