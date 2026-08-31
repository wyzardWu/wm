#!/bin/bash
# Action-axis eval. Usage:
#   bash eval_action.sh <run_name> <step>
# Reads ckpt from /home/zeqingwang/zeqingwang/models/final_model/sf2/baseline/<run_name>/step-<step>.safetensors
# Outputs to examples/ReactiveGWM/sf2/results/<run_name>_step<step>/action/
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

RUN_NAME=${1:?usage: eval_action.sh <run_name> <step>}
STEP=${2:?usage: eval_action.sh <run_name> <step>}

CKPT_ROOT="${CKPT_ROOT:-/home/zeqingwang/zeqingwang/models/train/sf2/Baseline/${RUN_NAME}}"
FULL_CKPT="${CKPT_ROOT}/step-${STEP}.safetensors"
LORA_CKPT="${CKPT_ROOT}/step-${STEP}_lora.safetensors"

# Pick whichever ckpt file exists.
if [ -f "${FULL_CKPT}" ]; then
  CKPT_FLAGS="--full_ckpt ${FULL_CKPT}"
elif [ -f "${LORA_CKPT}" ]; then
  CKPT_FLAGS="--lora_ckpt ${LORA_CKPT} --lora_alpha 0.8"
else
  echo "ERROR: ckpt not found at ${FULL_CKPT} or ${LORA_CKPT}"
  exit 1
fi

# Use one specific clip's first frame as the shared starting image.
REFERENCE_CLIP=${REFERENCE_CLIP:-"/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s/pure/00_noop_无操作/clip_0000/video.mp4"}
GPU_IDS=${GPU_IDS:-"0,1,2,3"}
OUT="examples/ReactiveGWM/sf2/results/${RUN_NAME}_step${STEP}/action"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

echo "=== SF2_final eval_action ==="
echo "Run:    ${RUN_NAME}  step=${STEP}"
echo "Ckpt:   ${CKPT_FLAGS}"
echo "Ref:    ${REFERENCE_CLIP}"
echo "Out:    ${OUT}"
echo "GPUs:   ${GPU_IDS}"

python examples/ReactiveGWM/inference/eval_action.py \
  --game sf2 \
  ${CKPT_FLAGS} \
  --reference_clip "${REFERENCE_CLIP}" \
  --output_dir "${OUT}" \
  --height 480 --width 608 --num_frames 101 \
  --num_inference_steps 30 \
  --gpu_ids "${GPU_IDS}"
