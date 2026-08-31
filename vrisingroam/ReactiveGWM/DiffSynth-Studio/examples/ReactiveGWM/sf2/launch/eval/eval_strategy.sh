#!/bin/bash
# Strategy-axis eval. Usage:
#   bash eval_strategy.sh <run_name> <step>
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

RUN_NAME=${1:?usage: eval_strategy.sh <run_name> <step>}
STEP=${2:?usage: eval_strategy.sh <run_name> <step>}

CKPT_ROOT="${CKPT_ROOT:-/home/zeqingwang/zeqingwang/models/train/sf2/Baseline/${RUN_NAME}}"
FULL_CKPT="${CKPT_ROOT}/step-${STEP}.safetensors"
LORA_CKPT="${CKPT_ROOT}/step-${STEP}_lora.safetensors"

if [ -f "${FULL_CKPT}" ]; then
  CKPT_FLAGS="--full_ckpt ${FULL_CKPT}"
elif [ -f "${LORA_CKPT}" ]; then
  CKPT_FLAGS="--lora_ckpt ${LORA_CKPT} --lora_alpha 0.8"
else
  echo "ERROR: ckpt not found at ${FULL_CKPT} or ${LORA_CKPT}"
  exit 1
fi

REFERENCE_CLIP=${REFERENCE_CLIP:-"/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s/pure/00_noop_无操作/clip_0000/video.mp4"}
GPU_IDS=${GPU_IDS:-"0,1,2,3"}
OUT="examples/ReactiveGWM/sf2/results/${RUN_NAME}_step${STEP}/strategy"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

echo "=== SF2_final eval_strategy ==="
echo "Run:    ${RUN_NAME}  step=${STEP}"
echo "Ckpt:   ${CKPT_FLAGS}"
echo "Ref:    ${REFERENCE_CLIP}"
echo "Out:    ${OUT}"
echo "GPUs:   ${GPU_IDS}"

python examples/ReactiveGWM/sf2/extras/eval_strategy.py \
  ${CKPT_FLAGS} \
  --reference_clip "${REFERENCE_CLIP}" \
  --output_dir "${OUT}" \
  --height 480 --width 608 --num_frames 101 \
  --num_inference_steps 30 \
  --gpu_ids "${GPU_IDS}"
