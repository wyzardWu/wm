#!/usr/bin/env bash
# Stage 3 DMD ckpt 离线批量 sanity 测试 — 便捷启动 (一次性, 跑完即止).
#
# 背景: Stage 3 训练机上 in-training sanity 全部 CUDA OOM (8 卡训练占满 GPU 0),
# samples/ 只有 .log 没有 mp4. 本脚本在一张空闲 GPU 上离线对现有 step-N.safetensors
# 批量出片 (idle + light_punch 各一个 60s 视频), 复用已验证的 sanity_sample.py.
#
# 用法:
#   bash examples/ReactiveGWM_casual_forcing/launch/test_stage3.sh            # 默认: 全部 ckpt, 自动挑卡
#   bash .../launch/test_stage3.sh --gpu 2                                    # 指定 GPU 2
#   bash .../launch/test_stage3.sh --steps latest                            # 只测最新一个 ckpt
#   bash .../launch/test_stage3.sh --steps 800 --fill_action light_punch     # 只测 step-800 的 lightpunch
#   bash .../launch/test_stage3.sh --dry_run                                 # 只打印将执行的命令
#   (建议后台跑: nohup bash .../launch/test_stage3.sh > /tmp/test_stage3.log 2>&1 & )
#
# Env overrides:
#   CONDA_ENV   conda 环境名 (默认 diffsynth_sf; 设空则不激活)
#   CKPT_DIR    覆盖 --ckpt_dir (默认指向 stage3_dmd 输出目录)
# 其余参数 (--steps/--gpu/--fill_action/--seed/--config/--out_dir/...) 直接透传给 python.

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

ARGS=("$@")
if [[ -n "${CKPT_DIR:-}" ]]; then
    ARGS+=(--ckpt_dir "${CKPT_DIR}")
fi

python -m examples.ReactiveGWM_casual_forcing.inference.test_stage3_ckpts "${ARGS[@]}"
