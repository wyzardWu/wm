#!/bin/bash
# SF2_ALL 3-model strategy evaluation. 270 jobs across GPUs 0-7.
# Resumable: re-running skips jobs whose <job_id>.json already exists.
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

OUT_DIR=${OUT_DIR:-/home/zeqingwang/zeqingwang/Paper_Figure/Strategy/SF2}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
ONLY_MODEL=${ONLY_MODEL:-}     # comma-list, e.g. "base"; empty = all 3

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

ARGS=(
  --output_dir "${OUT_DIR}"
  --gpu_ids    "${GPU_IDS}"
  --height 480 --width 608 --num_frames 101
  --num_inference_steps 30
  --cfg_scale 5.0
  --action_cfg_scale 1.0
)
if [ -n "${ONLY_MODEL}" ]; then
  ARGS+=(--only_model "${ONLY_MODEL}")
fi

echo "=== SF2_ALL strategy eval ==="
echo "Out:    ${OUT_DIR}"
echo "GPUs:   ${GPU_IDS}"
echo "Models: ${ONLY_MODEL:-all 3 (base, visual, visual_xattn_sf3)}"

python examples/ReactiveGWM/sf2/extras/eval_sf2all_strategy.py "${ARGS[@]}"
