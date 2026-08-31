#!/usr/bin/env bash
# Resume Stage 3 DMD from a state-<N>/ dir on 4 cards (GPU 0-3) — 8→4 FSDP reshard.
#
# 背景 (2026-05-24): 原 8 卡 FSDP run 在 ~step-4900 被杀 (tmux 会话丢失, 日志无 traceback);
# 现仅 GPU 0-3 空闲 (GPU 4-7 = danze 的活 OpenVLA job, 各 ~31GB / 70-93% util, 不可碰).
# 故改 4 卡 resume. 已知代价 (用户 2026-05-24 已确认):
#   - 全局 batch 8→4 (Stage 3 runner 固定 grad_accum=1, 全局 batch = 卡数) → 训练动态变化, 靠 sanity 盯;
#   - EMA shadow 重置: state-4800 无 ema-rank*.safetensors (跑出它的旧 runner 还没加持久化),
#     resume 后 EMA 从当前 generator 权重重累积 (decay 0.99, 约几百 gen-step 重收敛);
#     注意 step-4800.safetensors 本身导出的就是 EMA 权重 → 推理 EMA 快照未丢, 丢的只是平均历史;
#   - 单卡 FSDP 分片约翻倍 (8 卡时单卡 reserved ~102GB) → 有 OOM 风险. 0-3 全空, 爆了只崩自己 → 退回等卡.
# 拓扑校验: state-4800 无 cf_resume_meta.json → runner 的 _validate_resume_meta 走 legacy 分支
#   (只 warn 不 block), 8→4 放行, 不需要 dmd_allow_sharded_resume_mismatch.
# Gradient checkpointing 已在 default.yaml 开启 → 无额外保数学显存杠杆, 4 卡 OOM 即退回等卡.
#
# 用法 (建议 tmux 后台, 断连不丢):
#   TS=$(date +%Y%m%d_%H%M%S)
#   LOG=/home/zeqingwang/zeqingwang/models/train/sf3_casual_forcing/stage3_dmd/train_4card_resume4800_${TS}.log
#   tmux new-session -d -s cf_stage3 \
#     "bash examples/ReactiveGWM_casual_forcing/launch/resume_stage3_4card.sh 2>&1 | tee ${LOG}"
#
# Env 可覆盖: CUDA_VISIBLE_DEVICES / NUM_PROCESSES / RESUME_STATE / CONDA_ENV / CFG
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-diffsynth_sf}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export USE_FSDP=1
export RESUME_STATE="${RESUME_STATE:-/home/zeqingwang/zeqingwang/models/train/sf3_casual_forcing/stage3_dmd/state-4800}"

echo "[resume] $(date '+%F %T') CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_PROCESSES=${NUM_PROCESSES} USE_FSDP=${USE_FSDP}"
echo "[resume] RESUME_STATE=${RESUME_STATE}"
exec bash "${HERE}/stage3_dmd.sh"
