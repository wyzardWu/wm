#!/bin/bash
# Stage-3 Self-Forcing DMD on 252 (4 GPUs): student = CD ema-5000 (causal, 2-step),
# real-score teacher + critic init = noblock bidirectional teacher (clean distribution).
# F=26 adaptation: sink 3 + kv_window 22 = rope_cap 25 (= F-1 envelope; his F=21 defaults
# would be 5 frames OOD — his own BUG-2 class). VAE re-anchor via our wan_shim.
set -eo pipefail
cd ~/vrisingroam && . env.sh
cd ~/vrisingroam/cf_distill_3stage/stage3_dmd

GPUS="${GPUS:-2,3}" NPROC="${NPROC:-2}" PORT="${PORT:-29587}"
DATA=/data/yuzhewu/vrisingroam/distill/data/vrising_F26_v3_noblock0
TEACHER=/data/yuzhewu/vrisingroam/eval_ckpts/step-65000.safetensors
STUDENT=/data/yuzhewu/vrisingroam/distill/runs/gm_cd_v1/ema/ema_step5000.safetensors
OUT=/data/yuzhewu/vrisingroam/distill/runs/gm_dmd_v2
FSDP_CFG=../common/accelerate_fsdp_dmd.yaml

export GM_WAN_DIR=$HOME/vrisingroam/wan_shim
export GM_WAN_CKPT=/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B

echo "[stage3/252] student=$STUDENT teacher=$TEACHER -> $OUT"
CONDA_ENV="" CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_P2P_LEVEL=NVL OMP_NUM_THREADS=16 \
  accelerate launch --config_file "$FSDP_CFG" --num_processes "$NPROC" --main_process_port "$PORT" \
  dmd_gm.py \
    --data "$DATA" \
    --teacher_ckpt "$TEACHER" --student_ckpt "$STUDENT" \
    --out "$OUT" \
    --denoise_list 1000,250 \
    --kv_window 22 --sink_size 3 --rope_cap 25 \
    --rollout_min 26 --rollout_max 100 --dmd_score_frames 26 \
    --reanchor_every 16 --reanchor_win 6 \
    --critic_steps 4 --real_guidance_scale 1.0 \
    --lr 5e-6 --lr_critic 1e-6 --max_steps 6000 --warmup 100 \
    --opt8bit 1 --use_ema 0 --save_every 100 --keep_last_states 2 "$@"
