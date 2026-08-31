#!/bin/bash
# SF3 action-axis evaluation launcher.
#
# Usage:
#   bash examples/ReactiveGWM/sf3/inference/launch/eval_action.sh <run_name> <step>
# Optional env overrides:
#   GPU_IDS=0,1,2,3   bash ... <run_name> <step>
#   CFG_SCALE=5.0     bash ... <run_name> <step>   # only meaningful if trained w/ prompt_dropout
#   REFERENCE_CLIP=/path/to/video.mp4 bash ... <run_name> <step>
#   FULL_CKPT=/abs/path.safetensors   bash ... <run_name> <step>
#
# Output:
#   /opt/dlami/nvme/zeqingwang/evaluation/sf3/<run_name>_step<step>/action/
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <run_name> <step>"
  exit 1
fi
RUN_NAME="$1"
STEP="$2"

GPU_IDS="${GPU_IDS:-0}"
CFG_SCALE="${CFG_SCALE:-1.0}"
ACTION_CFG_SCALE="${ACTION_CFG_SCALE:-1.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
SEED="${SEED:-0}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-101}"

# Default checkpoint location: caller may override via FULL_CKPT.
FULL_CKPT="${FULL_CKPT:-/opt/dlami/nvme/zeqingwang/models/train/sf3/${RUN_NAME}/step-${STEP}.safetensors}"

# Default reference clip: caller MUST override via REFERENCE_CLIP for now (no
# stable default in test set yet — pass an absolute path to one clip's video.mp4).
REFERENCE_CLIP="${REFERENCE_CLIP:-}"
if [ -z "${REFERENCE_CLIP}" ]; then
  echo "ERROR: pass REFERENCE_CLIP=/path/to/clip/video.mp4 (its first frame is the eval input)."
  exit 1
fi
if [ ! -f "${REFERENCE_CLIP}" ]; then
  echo "ERROR: REFERENCE_CLIP not found: ${REFERENCE_CLIP}"
  exit 1
fi
if [ ! -f "${FULL_CKPT}" ]; then
  echo "ERROR: FULL_CKPT not found: ${FULL_CKPT}"
  exit 1
fi

OUT_DIR="/opt/dlami/nvme/zeqingwang/evaluation/sf3/${RUN_NAME}_step${STEP}/action"

echo "=== SF3 eval_action ==="
echo "Run:        ${RUN_NAME} step=${STEP}"
echo "Ckpt:       ${FULL_CKPT}"
echo "Reference:  ${REFERENCE_CLIP}"
echo "Output:     ${OUT_DIR}"
echo "GPUs:       ${GPU_IDS}"
echo "CFG:        prompt=${CFG_SCALE} action=${ACTION_CFG_SCALE}"
echo "Resolution: ${HEIGHT}x${WIDTH}, ${NUM_FRAMES} frames, ${NUM_INFERENCE_STEPS} steps"
echo "Start: $(date)"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

PYTHONUNBUFFERED=1 python examples/ReactiveGWM/inference/eval_action.py \
  --game sf3 \
  --full_ckpt "${FULL_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --reference_clip "${REFERENCE_CLIP}" \
  --gpu_ids "${GPU_IDS}" \
  --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --action_cfg_scale "${ACTION_CFG_SCALE}" \
  --seed "${SEED}"

echo "Done: $(date)"
echo "Results: ${OUT_DIR}"
