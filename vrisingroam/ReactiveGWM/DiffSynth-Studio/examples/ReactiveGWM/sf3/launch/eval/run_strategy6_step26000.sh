#!/bin/bash
# SF3 strategy-listening eval: 6 sampled clips (2 per category) from
# metadata_wo_pure_v5cat_10k.csv, run with each clip's CSV prompt so the
# Strategy(...) substring varies per job. cfg_scale=5.0 (training used
# prompt_dropout_prob=0.1).
#
# We launch 6 single-GPU python subprocesses in parallel, each pinned to a
# physical GPU via CUDA_VISIBLE_DEVICES set BEFORE the python process starts.
# Going through bash-level parallelism instead of mp.spawn is necessary because
# in-Python `os.environ["CUDA_VISIBLE_DEVICES"]` overrides inside the worker
# don't always take effect (observed all spawn workers landing on GPU 0).
#
# Coexists with running training on the same GPUs — inference peak ~13-15GB
# while training reserves ~60GB on 143GB cards.
set -e
cd /home/zeqingwang/zeqingwang/ReactiveGWM/DiffSynth-Studio

CKPT="${CKPT:-/home/zeqingwang/zeqingwang/models/train/sf3/SF3-model/p1_joint_480x832_5s_fixedprompt_coldstart_freeze_xattn/step-26000.safetensors}"
OUT_DIR="${OUT_DIR:-/home/zeqingwang/zeqingwang/evaluation/sf3/p1_step26000_strategy6}"
SAMPLES_CSV="${SAMPLES_CSV:-${OUT_DIR}/samples.csv}"
DATASET_BASE="${DATASET_BASE:-/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF3/train/clips_5s}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5}"
CFG_SCALE="${CFG_SCALE:-5.0}"
ACTION_CFG_SCALE="${ACTION_CFG_SCALE:-1.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi
if [ ! -f "${SAMPLES_CSV}" ]; then
  echo "ERROR: samples csv not found: ${SAMPLES_CSV}"
  exit 1
fi

IFS=',' read -ra GPU_IDS <<< "${GPU_IDS_CSV}"
SHARDS_DIR="${OUT_DIR}/_shards"
LOGS_DIR="${OUT_DIR}/_logs"
mkdir -p "${SHARDS_DIR}" "${LOGS_DIR}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate diffsynth

# Split samples.csv into 1-row shard CSVs (header + one data row each).
python - "${SAMPLES_CSV}" "${SHARDS_DIR}" <<'PY'
import sys, pandas as pd
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2])
df = pd.read_csv(src)
for i, (_, row) in enumerate(df.iterrows()):
    pd.DataFrame([row]).to_csv(dst / f"row_{i:02d}.csv", index=False)
print(f"[split] wrote {len(df)} shard CSVs to {dst}")
PY

NUM_ROWS=$(python -c "import pandas as pd; print(len(pd.read_csv('${SAMPLES_CSV}')))")
echo "=== SF3 strategy6 eval ==="
echo "Ckpt:         ${CKPT}"
echo "Samples CSV:  ${SAMPLES_CSV}  (${NUM_ROWS} rows)"
echo "Dataset base: ${DATASET_BASE}"
echo "Output:       ${OUT_DIR}"
echo "GPUs:         ${GPU_IDS_CSV}"
echo "cfg_scale:    ${CFG_SCALE}   action_cfg_scale: ${ACTION_CFG_SCALE}"
echo "Start:        $(date)"

PIDS=()
for i in $(seq 0 $((NUM_ROWS-1))); do
  GPU_IDX=$((i % ${#GPU_IDS[@]}))
  GPU="${GPU_IDS[$GPU_IDX]}"
  SHARD_CSV="${SHARDS_DIR}/row_$(printf '%02d' $i).csv"
  LOG="${LOGS_DIR}/row_$(printf '%02d' $i)_gpu${GPU}.log"
  echo "[launch] row $i -> physical GPU ${GPU}  log: ${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
    python examples/ReactiveGWM/sf3/extras/infer_sf3.py \
      --full_ckpt "${CKPT}" \
      --csv_path "${SHARD_CSV}" \
      --dataset_base "${DATASET_BASE}" \
      --output_dir "${OUT_DIR}" \
      --gpu_ids 0 \
      --height 480 --width 832 --num_frames 101 \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --cfg_scale "${CFG_SCALE}" \
      --action_cfg_scale "${ACTION_CFG_SCALE}" \
      --action_hold_window 10 \
      --use_csv_prompt \
      > "${LOG}" 2>&1 &
  PIDS+=($!)
done

echo "[launched] ${#PIDS[@]} workers, waiting ..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[fail] worker pid ${pid} exited non-zero"
    FAIL=$((FAIL+1))
  fi
done

echo "Done: $(date)"
echo "Failed workers: ${FAIL}"
echo "Results: ${OUT_DIR}"
exit ${FAIL}
