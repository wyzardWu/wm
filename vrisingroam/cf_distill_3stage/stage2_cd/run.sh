#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage-2 : Causal Consistency Distillation  (gm_cd_p1_tf_v4)  — the DEPLOY ckpt.
#
# Turns the many-step Stage-1 AR teacher into a few-step (2-step) causal student
# WITHOUT offline ODE-paired data: GT latents + the frozen teacher's single Euler
# step. P1 (faithful): teacher == student-init == the Stage-1 ckpt (师生同结构).
# CF generator-EMA weights are the deploy weights -> ema_step10000  (the kept ckpt;
# 2-step, bright, holds 30s — the strongest deploy candidate of the whole pipeline).
#
# Reproduces the run_p1_chain_watcher.sh STAGE-2 block -> run_cd_gm.sh chain.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$(dirname "$HERE")"
REPO="$(dirname "$TRAIN")"

CONDA_ENV="${CONDA_ENV:-gamemaster}"
[ -n "${CONDA_ENV}" ] && { source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV" || true; }

GPUS="${GPUS:-0,1,2,3}"; NPROC="${NPROC:-4}"; PORT="${PORT:-29572}"
DATA="${DATA:-$REPO/data/precomputed/all3_F21_full}"
# teacher AND student-init are BOTH the Stage-1 AR ckpt (P1 same-trajectory)
AR_CKPT="${AR_CKPT:-$REPO/ckpt/stage1_ar_teacher/gm_ar_tf_v4_step20000.safetensors}"
OUT="${OUT:-$REPO/ckpt/stage2_cd/runs/gm_cd_p1_tf_v4}"
FSDP_CFG="${FSDP_CFG:-$TRAIN/common/accelerate_fsdp_dmd.yaml}"   # 3-model FSDP (student+ema+teacher)

echo "[stage2] AR teacher/student=$AR_CKPT -> out=$OUT"
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
  accelerate launch --config_file "$FSDP_CFG" --num_processes "$NPROC" --main_process_port "$PORT" \
  "$HERE/cd_gm.py" \
    --data "$DATA" \
    --teacher_ckpt "$AR_CKPT" --student_ckpt "$AR_CKPT" --teacher_causal 1 \
    --teacher_forcing --windowed_train 20 --windowed_sink 1 \
    --discrete_cd_N 48 --real_guidance_scale 1.0 \
    --lr 2e-6 --max_steps 12000 --warmup 100 \
    --ema_decay 0.99 --ema_start 200 \
    --save_every 500 --keep_last_states 2 --out "$OUT" "$@"
# Deploy weight = $OUT/ema/ema_step10000.safetensors  (the EMA file, NOT dits/dit_step*).
# Inference: 2-step, --denoise_list 1000,250 (must end low-noise), CFG=1.0 iron rule.
