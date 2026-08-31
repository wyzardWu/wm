#!/usr/bin/env bash
# Stage 1 AR ckpt 离线批量 sanity 测试 — 便捷启动 (一次性, 跑完即止).
#
# 背景: Stage 1 aligned 重训 (sf3_casual_forcing_2/stage1_ar) 的 in-training sanity
# 在 4 卡训练机上全部 CUDA OOM —— 训练把 GPU0 占到 ~73GB, spawn 的 sanity subprocess
# 受 cfg.sanity_sample_memory_fraction=0.2 (≈28GB) 上限卡死, DiT 搬不上卡.
# samples/ 里只有 step_N.log (OOM traceback), 一个 mp4 都没有.
#
# 本脚本复用通用引擎 test_stage3_ckpts.py (名字叫 stage3, 逻辑 stage-agnostic), 在一张
# 共享卡上离线对现有 step-N.safetensors 批量出片. 关键区别:
#   - --ckpt_dir / --config 默认指向 stage1 (而非 stage3_dmd)
#   - --memory_fraction 0.35 (≈49GB): 共享训练卡有 ~66GB free, 0.2 太紧会 OOM;
#     0.35 + 训练 73GB = 122GB < 140GB, 不会拖垮训练 (但 100% util 下会拖慢, 视频很慢)
#   - Stage1 是 50 步 diffusion (config 决定), 单个 60s 视频远比 Stage3 的 4 步慢,
#     整批 (5 ckpt × idle/light_punch) 可能数小时 → 建议 nohup 后台跑.
#
# 用法:
#   nohup bash examples/ReactiveGWM_casual_forcing/launch/test_stage1.sh \
#       > /tmp/test_stage1.log 2>&1 &        # 默认: 全部 ckpt, auto 挑最空卡
#   bash .../launch/test_stage1.sh --steps latest                   # 只测最新 ckpt
#   bash .../launch/test_stage1.sh --steps 10000 --gpu 0            # 指定 step + 卡
#   bash .../launch/test_stage1.sh --dry_run                       # 只打印命令
#
# Env overrides:
#   CONDA_ENV        conda 环境名 (默认 diffsynth_sf; 设空则不激活)
#   CKPT_DIR         覆盖 ckpt 目录 (默认 sf3_casual_forcing_2/stage1_ar)
#   STAGE1_CONFIG    覆盖 config (默认 configs/stage1_ar.yaml)
#   MEM_FRACTION     覆盖 --memory_fraction (默认 0.35)
# 其余参数 (--steps/--gpu/--fill_action/--seed/--out_dir/...) 直接透传给 python.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-diffsynth_sf}"
if [[ -n "${CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_DIR="${CKPT_DIR:-/home/zeqingwang/zeqingwang/models/train/sf3_casual_forcing_2/stage1_ar}"
STAGE1_CONFIG="${STAGE1_CONFIG:-${HERE}/configs/stage1_ar.yaml}"
MEM_FRACTION="${MEM_FRACTION:-0.35}"

ARGS=(--ckpt_dir "${CKPT_DIR}" --config "${STAGE1_CONFIG}" --memory_fraction "${MEM_FRACTION}")
ARGS+=("$@")

python -m examples.ReactiveGWM_casual_forcing.inference.test_stage3_ckpts "${ARGS[@]}"
