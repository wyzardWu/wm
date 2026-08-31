#!/bin/bash
# SF3 evaluation on the test split — step-26000 checkpoint, prompt cfg=5.0.
#
# Validates action conditioning + strategy/prompt response on the held-out
# 120-clip test set using the cold-start fixed-prompt phase-1 run.
#
# Override via env:
#   GPU_IDS=0,1,2,3   bash run_eval_step26000_cfg5.sh
#   MAX_CLIPS=8       bash run_eval_step26000_cfg5.sh   # smoke test
#   USE_CSV_PROMPT=1  bash run_eval_step26000_cfg5.sh   # per-clip NPC prompt
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

CKPT="${CKPT:-/home/zeqingwang/zeqingwang/models/train/sf3/SF3-model/p1_joint_480x832_5s_fixedprompt_coldstart_freeze_xattn/step-26000.safetensors}"
OUT_DIR="${OUT_DIR:-/home/zeqingwang/zeqingwang/evaluation/sf3/p1_fixedprompt_step26000_cfg5}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
CFG_SCALE="${CFG_SCALE:-5.0}"
ACTION_CFG_SCALE="${ACTION_CFG_SCALE:-1.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
MAX_CLIPS_FLAG=""
if [ -n "${MAX_CLIPS}" ]; then
  MAX_CLIPS_FLAG="--max_clips ${MAX_CLIPS}"
fi
PROMPT_FLAG=""
if [ -n "${USE_CSV_PROMPT}" ] && [ "${USE_CSV_PROMPT}" != "0" ]; then
  PROMPT_FLAG="--use_csv_prompt"
fi

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

echo "=== SF3 eval ==="
echo "Ckpt:      ${CKPT}"
echo "Output:    ${OUT_DIR}"
echo "GPUs:      ${GPU_IDS}"
echo "cfg_scale: ${CFG_SCALE}   action_cfg_scale: ${ACTION_CFG_SCALE}"
echo "steps:     ${NUM_INFERENCE_STEPS}   prompt: ${PROMPT_FLAG:-fixed}"
echo "Start: $(date)"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

PYTHONUNBUFFERED=1 python examples/ReactiveGWM/sf3/extras/infer_sf3.py \
  --full_ckpt "${CKPT}" \
  --output_dir "${OUT_DIR}" \
  --gpu_ids "${GPU_IDS}" \
  --height 480 --width 832 --num_frames 101 \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --action_cfg_scale "${ACTION_CFG_SCALE}" \
  --action_hold_window 10 \
  ${PROMPT_FLAG} \
  ${MAX_CLIPS_FLAG}

echo "Done: $(date)"
echo "Results: ${OUT_DIR}"
