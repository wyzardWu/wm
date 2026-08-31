#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 : AR-diffusion TEACHER  (gm_ar_tf_v4)  — the CF++ conditioning fix.
#
# Causalize the bidirectional teacher gm_all3_v1 into a many-step causal AR-diffusion
# model trained with TEACHER FORCING (clean_x channel + TF attn mask), NOT diffusion-
# forcing. This is the single change that lets the model autoregressively roll out 30s
# without exposure-bias collapse.  Produces -> dit_step20000  (the kept Stage-1 ckpt).
#
# Reproduces the original launchers/run_ar_tf_v4.sh -> run_ar_diffusion_gm.sh chain.
# Override anything via env vars; defaults point at this repo's layout.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$(dirname "$HERE")"
REPO="$(dirname "$TRAIN")"

# --- environment (override CONDA_ENV / leave blank to use the active env) ---
CONDA_ENV="${CONDA_ENV:-gamemaster}"
[ -n "${CONDA_ENV}" ] && { source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV" || true; }

# --- inputs (EDIT these to your data / base teacher) ---
GPUS="${GPUS:-0,1,2,3}"; NPROC="${NPROC:-4}"; PORT="${PORT:-29571}"
DATA="${DATA:-$REPO/data/precomputed/all3_F21_full}"                       # full 23,896-clip set
INIT="${INIT:-$REPO/../vrising_dataset/ckpt/gm_all3_v1/dit_step30000.safetensors}"  # bidir base teacher
OUT="${OUT:-$REPO/ckpt/stage1_ar_teacher/runs/gm_ar_tf_v4}"
FSDP_CFG="${FSDP_CFG:-$TRAIN/common/accelerate_fsdp.yaml}"

echo "[stage1] init=$INIT data=$DATA -> out=$OUT"
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
  accelerate launch --config_file "$FSDP_CFG" --num_processes "$NPROC" --main_process_port "$PORT" \
  "$HERE/ar_diffusion_gm.py" \
    --data "$DATA" --init_ckpt "$INIT" --out "$OUT" \
    --teacher_forcing --tf_noise_aug_max_idx 0 \
    --windowed_train 0 --windowed_sink 0 \
    --lr 1e-5 --grad_accum 2 --max_steps 20000 --warmup 200 \
    --save_every 500 --keep_last_states 2 --resume auto "$@"
