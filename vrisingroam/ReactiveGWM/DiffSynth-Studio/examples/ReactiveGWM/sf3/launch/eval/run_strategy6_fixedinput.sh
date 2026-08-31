#!/bin/bash
# Strategy listening eval with shared first-frame + zero-action.
#
# All 6 strategy prompts are run against the same reference clip's first frame
# (default: control_balance_distance.mp4 from p1_step26000_strategy6) with an
# all-zero keyboard action. Only the prompt's `Strategy(...)` substring varies
# across jobs — single-variable test of strategy listening.
#
# 6 strategies are split across `${GPU_IDS}` GPUs. Each GPU runs its assigned
# slugs sequentially in one process (model loaded once). CUDA_VISIBLE_DEVICES
# is set before python so workers reliably bind to their assigned physical GPU.
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

CKPT="${CKPT:?must set CKPT}"
OUT_DIR="${OUT_DIR:?must set OUT_DIR}"
SAMPLES_CSV="${SAMPLES_CSV:-/home/zeqingwang/zeqingwang/evaluation/sf3/p1_step26000_strategy6/samples.csv}"
REFERENCE_VIDEO="${REFERENCE_VIDEO:-/home/zeqingwang/zeqingwang/evaluation/sf3/p1_step26000_strategy6/control_balance_distance.mp4}"
GPU_IDS_CSV="${GPU_IDS:-4,5,6,7}"
CFG_SCALE="${CFG_SCALE:-5.0}"
ACTION_CFG_SCALE="${ACTION_CFG_SCALE:-1.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"

# Resolve the reference video (may be a symlink).
REFERENCE_VIDEO="$(readlink -f "${REFERENCE_VIDEO}")"

if [ ! -f "${CKPT}" ]; then echo "ERROR: ckpt not found: ${CKPT}"; exit 1; fi
if [ ! -f "${SAMPLES_CSV}" ]; then echo "ERROR: samples csv not found: ${SAMPLES_CSV}"; exit 1; fi
if [ ! -f "${REFERENCE_VIDEO}" ]; then echo "ERROR: reference video not found: ${REFERENCE_VIDEO}"; exit 1; fi

mkdir -p "${OUT_DIR}"
LOGS_DIR="${OUT_DIR}/_logs"
mkdir -p "${LOGS_DIR}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

# Compute slugs from samples_csv and split round-robin across GPU_IDS.
read -r -a GPU_IDS <<< "$(echo "${GPU_IDS_CSV}" | tr ',' ' ')"
NGPU=${#GPU_IDS[@]}
mapfile -t SLUGS < <(python - "${SAMPLES_CSV}" <<'PY'
import sys, pandas as pd
SLUGS = {
    "Offense: Closes the distance quickly to apply pressure and initiate close combat.": "offense_close_distance",
    "Offense: Maintains constant aggression to overwhelm the opponent and force defensive reactions.": "offense_constant_aggression",
    "Offense: Focuses on advancing and chaining attacks to keep the opponent on the back foot.": "offense_advance_chain",
    "Defense: Holds ground with blocks and reactive counters, only striking when an opening appears.": "defense_hold_ground",
    "Defense: Prioritizes guarding and reading the opponent's actions over initiating offense.": "defense_guard_read",
    "Defense: Absorbs and evades incoming pressure, recovering safely instead of trading hits.": "defense_absorb_evade",
    "Control: Manages spacing with projectiles and measured pokes to dictate the pace of engagement.": "control_spacing_projectiles",
    "Control: Balances offense and defense by controlling distance, neither rushing in nor purely turtling.": "control_balance_distance",
    "Control: Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.": "control_zoning_range",
}
df = pd.read_csv(sys.argv[1])
for _, r in df.iterrows():
    s = SLUGS.get(r["strategy"].strip())
    if s: print(s)
PY
)
echo "[plan] ${#SLUGS[@]} slugs across ${NGPU} GPUs"

declare -A GPU_TO_SLUGS
for i in "${!SLUGS[@]}"; do
  g="${GPU_IDS[$((i % NGPU))]}"
  if [ -n "${GPU_TO_SLUGS[$g]}" ]; then
    GPU_TO_SLUGS[$g]="${GPU_TO_SLUGS[$g]},${SLUGS[$i]}"
  else
    GPU_TO_SLUGS[$g]="${SLUGS[$i]}"
  fi
done

echo "=== fixed-input strategy eval ==="
echo "Ckpt:          ${CKPT}"
echo "Reference:     ${REFERENCE_VIDEO}"
echo "Output:        ${OUT_DIR}"
echo "GPUs:          ${GPU_IDS_CSV}"
echo "cfg_scale:     ${CFG_SCALE}    action_cfg_scale: ${ACTION_CFG_SCALE}"
echo "Start: $(date)"

PIDS=()
for g in "${GPU_IDS[@]}"; do
  SL="${GPU_TO_SLUGS[$g]}"
  if [ -z "${SL}" ]; then continue; fi
  LOG="${LOGS_DIR}/gpu${g}.log"
  echo "[launch] gpu=${g} slugs=${SL}  log=${LOG}"
  CUDA_VISIBLE_DEVICES="${g}" PYTHONUNBUFFERED=1 \
    python examples/ReactiveGWM/sf3/extras/infer_strategy_fixedinput.py \
      --full_ckpt "${CKPT}" \
      --reference_video "${REFERENCE_VIDEO}" \
      --samples_csv "${SAMPLES_CSV}" \
      --slugs "${SL}" \
      --output_dir "${OUT_DIR}" \
      --height 480 --width 832 --num_frames 101 \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --cfg_scale "${CFG_SCALE}" \
      --action_cfg_scale "${ACTION_CFG_SCALE}" \
      > "${LOG}" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then FAIL=$((FAIL+1)); fi
done
echo "Done: $(date)  Failed: ${FAIL}"
echo "Results: ${OUT_DIR}"
exit ${FAIL}
