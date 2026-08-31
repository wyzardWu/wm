#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage-3 : Self-Forcing DMD  (gm_dmd_p1cf_v5)  — CF++-aligned, structure-fixed.
#
# Distribution-Matching Distillation with autoregressive (KV-cache) self-forcing
# rollout. real-score + critic-init = the bidir teacher gm_all3_v1. Student-init =
# the Stage-2 CD ema. This is the v5 CF++-ALIGNED retrain that fixed the 4 bugs of
# the old collapsing gm_dmd_p1ra/p1_tf_v4 run:
#   #1 dmd_score_frames 26->20   (keep score window inside teacher F=21 -> pred_real in-dist)
#   #2 rollout 26-100 -> 21-27, grad_horizon 14->20   (kill deep unsupervised drift)
#   #3 reanchor_every 16 -> 0     (drop live-cache VAE writeback; VAE build auto-skips)
#   #4 critic_steps 4 -> 5        (match CF++ dfake_gen_update_ratio=5)
# Produces -> dit_step250  (the kept Stage-3 ckpt; structure fixed, residual darkening
# 48->40 OPEN, so NOT the deploy pick — Stage-2 CD is. Kept for the record / further work).
#
# Reproduces the original launchers/p1_dmd_cf_v5.sh -> run_dmd_gm.sh chain.
# NOTE: this is the cf_v5 launcher, NOT the chain-watcher DMD block (that was the OLD
# collapsing p1ra/p1_tf_v4 geometry kv20/rope21/sink1/score26 — deliberately dropped).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="$(dirname "$HERE")"
REPO="$(dirname "$TRAIN")"

CONDA_ENV="${CONDA_ENV:-gamemaster}"
[ -n "${CONDA_ENV}" ] && { source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV" || true; }
mkdir -p "${TMPDIR:-/tmp}"

GPUS="${GPUS:-0,1,2}"; NPROC="${NPROC:-3}"; PORT="${PORT:-29576}"   # DMD = 3-model (gen+critic+teacher)
DATA="${DATA:-$REPO/data/precomputed/all3_F21_full}"
TEACHER="${TEACHER:-$REPO/../vrising_dataset/ckpt/gm_all3_v1/dit_step30000.safetensors}"     # bidir real-score teacher
STUDENT="${STUDENT:-$REPO/ckpt/stage2_cd/gm_cd_p1_tf_v4_ema_step10000.safetensors}"          # CD ema generator-init
OUT="${OUT:-$REPO/ckpt/stage3_dmd/runs/gm_dmd_p1cf_v5}"
FSDP_CFG="${FSDP_CFG:-$TRAIN/common/accelerate_fsdp_dmd.yaml}"
# Wan base only needed if reanchor is re-enabled (off here): GM_WAN_DIR / GM_WAN_CKPT
export GM_WAN_CKPT="${GM_WAN_CKPT:-$REPO/../wan_base/Wan2.2-TI2V-5B}"

echo "[stage3] teacher=$TEACHER student(CD ema)=$STUDENT -> out=$OUT"
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
  accelerate launch --config_file "$FSDP_CFG" --num_processes "$NPROC" --main_process_port "$PORT" \
  "$HERE/dmd_gm.py" \
    --data "$DATA" \
    --teacher_ckpt "$TEACHER" --student_ckpt "$STUDENT" --out "$OUT" \
    --denoise_list 1000,250 --real_guidance_scale 1.0 \
    --kv_window 17 --rope_cap 20 --sink_size 3 \
    --rollout_min 21 --rollout_max 27 --dmd_score_frames 20 --grad_horizon 20 \
    --reanchor_every 0 --critic_steps 5 \
    --lr 2e-6 --lr_critic 1e-6 --max_steps 300 --warmup 100 \
    --use_ema 1 --ema_decay 0.99 --ema_start 100 \
    --save_every 50 --keep_last_states 2 "$@"
# ── FIX (2026-06-28): lr 5e-6 -> 2e-6 + deploy the EMA. The old lr_g=5e-6 (2.5x CF++'s 2e-6)
#    caused a high-LR min-max COLLAPSE: bright+structured for ~100 steps then a sudden dark
#    collapse (~step100->150). lr 2e-6 holds bright+stable; the EMA(0.99) deploy ckpt is
#    GT-brightness-matched, holds 30s, and BEATS CD at minute-scale (CD drifts ~52 at 105-120s,
#    DMD-EMA stays ~48). Deploy ckpt = ema_step300 -> ../ckpt/stage3_dmd/gm_dmd_lr2e6_ema_step300.
#    Eval/diag: brightness_eval.py / diag_pull.py / diag_grad.py / make_video.py. NOT a code bug;
#    nobody had a per-50-step brightness curve so they deployed the post-collapse step250.
# (old broken recipe was: --lr 5e-6 --max_steps 5000 --save_every 250  (no EMA))
