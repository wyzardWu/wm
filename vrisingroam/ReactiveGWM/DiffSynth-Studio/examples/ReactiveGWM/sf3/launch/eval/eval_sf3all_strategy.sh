#!/bin/bash
# SF3_ALL 3-model strategy evaluation. 270 jobs across GPUs 0-7.
# Resumable: re-running skips jobs whose .json already exists (flat or organized).
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

OUT_DIR=${OUT_DIR:-/home/zeqingwang/zeqingwang/Paper_Figure/Strategy/SF3}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
ONLY_MODEL=${ONLY_MODEL:-}

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

ARGS=(
  --output_dir "${OUT_DIR}"
  --gpu_ids    "${GPU_IDS}"
  --height 480 --width 832 --num_frames 101
  --num_inference_steps 30
  --cfg_scale 5.0
  --action_cfg_scale 1.0
)
[ -n "${ONLY_MODEL}" ] && ARGS+=(--only_model "${ONLY_MODEL}")

echo "=== SF3_ALL strategy eval ==="
echo "Out:    ${OUT_DIR}"
echo "GPUs:   ${GPU_IDS}"
echo "Models: ${ONLY_MODEL:-all 3 (base, visual, visual_xattn_sf2)}"

python examples/ReactiveGWM/sf3/extras/eval_sf3all_strategy.py "${ARGS[@]}"
