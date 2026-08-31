#!/usr/bin/env bash
# 三 stage 主 launcher 共享: 路径/环境 setup + --stage/--config/--resume_state 构建。
# 不抽 `accelerate launch` 本身 —— 三 stage 调用形态差异大（Stage 3 有 LAUNCH_FLAGS 数组
# + FSDP 分支 + PYTORCH_CUDA_ALLOC_CONF），抽进来会把 Stage 3 特殊性挤进通用脚本。
#
# 用法（在 launcher 顶部）:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#   CFG="${CFG:-${HERE}/configs/<stage>.yaml}"
#   ARGS=(); cf_build_args <stage> "${CFG}"
#   accelerate launch ... "${TRAIN_PY}" "${ARGS[@]}"
#
# source 后导出: HERE (example 目录) / REPO_ROOT / TRAIN_PY，并 cd 到 REPO_ROOT +
# 设好 PYTHONPATH / TOKENIZERS_PARALLELISM / PYTHONUNBUFFERED。

set -euo pipefail

# BASH_SOURCE[1] = 调用者 (各 stage launcher), 不是 _common.sh 本身。
HERE="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
TRAIN_PY="${HERE}/train.py"

cf_build_args() {
    # 用法: ARGS=(); cf_build_args <stage> <cfg_path>
    # 把 --stage X --config Y [--resume_state Z] 推入调用者作用域的 ARGS 数组。
    ARGS+=(--stage "$1" --config "$2")
    if [[ -n "${RESUME_STATE:-}" ]]; then
        ARGS+=(--resume_state "${RESUME_STATE}")
    fi
}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
