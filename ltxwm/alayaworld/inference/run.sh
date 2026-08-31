#!/usr/bin/env bash
# Alaya World — official image-to-video inference launcher.
#
# One command turns a "case" into a ~1-minute video (with a one-off skill effect
# in the final seconds). A CASE is just a folder of files sharing one <prefix>:
#
#     <prefix>_image.<png|jpg|jpeg|webp|bmp>   first frame that seeds the clip
#     <prefix>_camera.pt                       camera trajectory (cam_c2w [F,4,4] + intrinsic)
#     <prefix>_prompt.txt                      text prompt (the whole clip)
#     <prefix>_skill.txt        (optional)     prompt for the final --skill-sec seconds
#     ( or <prefix>_video.mp4 instead of _image.* to seed from a real video )
#
# The bundled reference case is playground/case1/  ->  prefix "playground/case1/case1".
# Model / weight paths live in the --cfg yaml (configs/infer.yaml) under `paths:`,
# NOT here. Everything you tune per-run is the config block below (override any of
# it from the environment) plus flags forwarded to `inference.run`.
#
# Examples:
#     bash inference/run.sh                                  # the bundled case1 (single GPU, ~1 min)
#     CASE=playground/case2/case2 bash inference/run.sh      # a different case
#     GPUS=4 bash inference/run.sh                           # Context Parallel across 4 GPUs
#     CUDA_VISIBLE_DEVICES=0 bash inference/run.sh           # pin to one GPU
#     bash inference/run.sh --seed 1234 --rounds 45          # extra flags -> inference.run
#     bash inference/run.sh --skill-sec 0                    # disable the end skill effect
#
# All flags after the script name are forwarded to `inference.run`
# (see: python -m inference.run --help).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# ============================== configuration ==============================
CASE=${CASE:-playground/case1/case1}   # which case to run (file prefix); default = case1
CFG=${CFG:-configs/infer.yaml}         # inference config; model/weight paths live under paths:
GPUS=${GPUS:-1}                        # 1 = single GPU; >1 = Context Parallel (N in {2, 4})
# ===========================================================================

export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

# Use $CASE as --input, and $CFG as --cfg, unless the caller passed them explicitly
# (an explicit flag on the command line wins — argparse takes the last occurrence).
has_input=0
for a in "$@"; do [[ "$a" == "--input" || "$a" == --input=* ]] && has_input=1; done
[[ "$has_input" -eq 0 ]] && set -- --input "$CASE" "$@"
set -- --cfg "$CFG" "$@"

if [[ "$GPUS" -gt 1 ]]; then
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
    python -m torch.distributed.run --nproc_per_node="$GPUS" -m inference.run "$@"
else
  python -m inference.run "$@"
fi
