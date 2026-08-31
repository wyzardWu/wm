#!/usr/bin/env bash
# Stage 3 DMD launch — multi-GPU accelerate DDP (OOM 切 FSDP 单 unit).
#
# Stage 3 = 3×5B 同驻 (generator 因果 + critic 双向可训 + teacher 双向冻结) + 4-step
# self-rollout + 双向 DMD backward, 显存远大于 Stage 1/2. full run 需充足空闲显存.
#
# Env overrides:
#   BASE_CKPT      override cfg.base_ckpt (teacher/critic 双向 base 起点)
#   DATA_ROOT      override cfg.dataset_base
#   OUT            override cfg.output_path
#   STUDENT_INIT   override cfg.student_init (generator 起点 = Stage 2 CD EMA 产物)
#   RESUME_STATE   path to a `state-<N>/` dir (含 ema.safetensors) to resume from
#   CFG            override config yaml (默认 configs/stage3_dmd.yaml)
#   NUM_PROCESSES  进程数 = 全局 batch (默认 8; 4-card 用 4; Stage 3 runner 固定 grad_accum=1)
#   MAIN_PORT      主进程端口 (默认 29522; 与 Stage 1/2 错开)
#   USE_FSDP       =1 切 FSDP 单 unit 兜底 (OOM 时; 需 dry-run 验证, 默认 DDP)
#
# 4-card dry-run (3.2.7, 指向真实 Stage 2 ckpt):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 \
#     CFG=$(pwd)/examples/ReactiveGWM_casual_forcing/configs/stage3_dmd_dryrun.yaml \
#     STUDENT_INIT=/.../stage2_cd/step-XXXXX.safetensors \
#     bash examples/ReactiveGWM_casual_forcing/launch/stage3_dmd.sh

set -euo pipefail

# 共享 path/env setup + cf_build_args (见 _common.sh)。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CFG="${CFG:-${HERE}/configs/stage3_dmd.yaml}"

ARGS=()
cf_build_args dmd "${CFG}"

# 抗显存碎片 (3×5B + rollout 反复分配 KV-cache / 激活) —— Stage 3 专有。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LAUNCH_FLAGS=(
    --num_processes "${NUM_PROCESSES:-8}"
    --num_machines 1
    --mixed_precision "${MIXED_PRECISION:-no}"
    --main_process_port "${MAIN_PORT:-29522}"
)
if [[ "${USE_FSDP:-0}" == "1" ]]; then
    # 冻结的 T5/VAE 不参与训练；从 FSDP root flat-param 中排除可显著降低 NO_WRAP
    # all-gather 大小，避免 critic-only step 因 30.52GB full flat-param gather 临界 OOM。
    export FSDP_IGNORED_MODULES="${FSDP_IGNORED_MODULES:-pipe.text_encoder|pipe.vae}"
    # FSDP NO_WRAP / FULL_SHARD: Stage 3 的 generator rollout 会反复调用 forward、
    # refill_kv_cache、_patchify_and_geom 等多入口 KV-cache 方法；per-block/per-model
    # auto-wrap 会让这些非 __call__ 入口读到 FSDP 分片参数并触发 conv/shape/dtype 错误。
    # 因此 FSDP 兜底保持单 unit NO_WRAP；显存用首帧 VAE 修复 + empty_cache + dry-run
    # 诊断控制，不改变 DMD 数学语义。
    ARGS+=(--use_fsdp)
    LAUNCH_FLAGS+=(
        --use_fsdp
        --fsdp_sharding_strategy FULL_SHARD
        --fsdp_use_orig_params true
        --fsdp_state_dict_type SHARDED_STATE_DICT
    )
else
    LAUNCH_FLAGS+=(--multi_gpu)
fi

accelerate launch "${LAUNCH_FLAGS[@]}" "${TRAIN_PY}" "${ARGS[@]}"
