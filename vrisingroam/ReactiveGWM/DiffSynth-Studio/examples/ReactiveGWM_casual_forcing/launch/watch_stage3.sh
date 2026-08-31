#!/usr/bin/env bash
# Stage 3 DMD ckpt 定时检测器 — 每 INTERVAL 秒扫描一次, 发现未测过的新 ckpt 就自动 sanity 测试.
#
# 增量逻辑全在 test_stage3_ckpts.py:
#   --steps auto      每轮重新列目录 (keep_last_ckpts 轮换删旧的也能跟上)
#   --skip_existing   已有 mp4 的 (step,fill) 跳过 → 只测新出现的
#   --retries 2       单个视频 OOM/失败自动换最空的卡重试 (共享机器竞态自愈)
# 循环串行 (跑完一轮才 sleep), 天然无重叠; 一轮没新 ckpt 就是秒级 no-op.
#
# 用法 (前台, Ctrl-C 停):
#   bash examples/ReactiveGWM_casual_forcing/launch/watch_stage3.sh
# 推荐 (后台脱离终端, 写 PID 便于停):
#   setsid nohup bash examples/ReactiveGWM_casual_forcing/launch/watch_stage3.sh \
#       > /tmp/watch_stage3.log 2>&1 < /dev/null &
#   echo $! > /tmp/watch_stage3.pid          # 停: kill $(cat /tmp/watch_stage3.pid)
#
# Env overrides:
#   INTERVAL    扫描间隔秒 (默认 300 = 5 分钟)
#   RETRIES     单视频失败重试次数 (默认 2)
#   CONDA_ENV   conda 环境名 (默认 diffsynth_sf; 设空则不激活)
#   CKPT_DIR    覆盖 ckpt 目录 (默认 stage3_dmd 输出目录)
# 其余 (--fill_action/--config/--out_dir/--seed/...) 直接透传给 python.

set -uo pipefail   # 不用 -e: 某轮部分视频失败 (返回 1) 不应终止整个 watcher

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

INTERVAL="${INTERVAL:-300}"
RETRIES="${RETRIES:-2}"

EXTRA=("$@")
if [[ -n "${CKPT_DIR:-}" ]]; then
    EXTRA+=(--ckpt_dir "${CKPT_DIR}")
fi

echo "[watch] 启动: 每 ${INTERVAL}s 扫描一次 stage3 新 ckpt; retries=${RETRIES}; $(date '+%F %T')"
while true; do
    echo "[watch] ---- 扫描 $(date '+%F %T') ----"
    python -m examples.ReactiveGWM_casual_forcing.inference.test_stage3_ckpts \
        --steps auto --skip_existing --retries "${RETRIES}" "${EXTRA[@]}" \
        || echo "[watch] 本轮非全成功 (部分失败, 下轮 skip_existing 会自动重测未出片的)"
    echo "[watch] 本轮结束, sleep ${INTERVAL}s"
    sleep "${INTERVAL}"
done
