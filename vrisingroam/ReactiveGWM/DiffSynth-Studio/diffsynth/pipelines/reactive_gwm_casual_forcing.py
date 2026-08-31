"""ReactiveGWM Causal Forcing pipeline — Stage 1/2/3 共用 model_fn + 调度 helpers.

自包含, 与 `diffsynth/pipelines/reactive_gwm_self_forcing.py` 文件并列但**互不引用**.

Stage 1 (本文件本阶段实现):
- `build_cf_scheduler` / `sample_uniform_timestep`: FlowMatchScheduler 复用 + 单 uniform t 采样
- `add_noise_per_frame`: per-frame timestep 加噪 (frame 0 anchor 用)
- `model_fn_causal_forcing(tf_mode=True / False, ...)`: 两路径 dispatch
- `sample_multistep_kv_cache`: sanity 推理用 (KV-cache 自回归 + multi-step per frame)

Stage 2 追加: `cd_teacher_one_step` (CFG ODE 推进)
Stage 3 追加: `holistic_dmd_helpers` (teacher CFG score / critic flow / normalizer / loss_gen)
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from ..diffusion import FlowMatchScheduler
from .reactive_gwm import model_fn_wan_video


# Wan 默认负面串, 与 CF 上游 `causal_cd_framewise.yaml` 的 negative_prompt 逐字一致.
# 放 pipeline 模块作单一来源: Stage 2 CD (`modules/cd.py`) 与 sanity 推理
# (`inference/sanity_sample.py`) 都从这里 import, 避免字符串漂移.
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


# --------------------------------------------------------------------- scheduler
def build_cf_scheduler(
    num_train_timestep: int = 1000,
    sigma_shift: float = 5.0,
    training: bool = True,
) -> FlowMatchScheduler:
    """配置一个 Wan-flavor FlowMatchScheduler, 1000 步离散.

    与 `diffsynth.pipelines.reactive_gwm.ReactiveGWMPipeline.scheduler` 默认行为一致
    (`shift=5.0`, `sigma_min=0.0`, `template='Wan'`); 训练时调用方设置 `training=True`
    使其计算 BSMNTW 权重并标记 training mode.
    """
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(
        num_inference_steps=num_train_timestep,
        denoising_strength=1.0,
        training=training,
        shift=sigma_shift,
    )
    return sched


def sample_uniform_timestep(
    scheduler: FlowMatchScheduler,
    batch_size: int,
    min_step: int,
    max_step: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """单 uniform t ∈ [min_step, max_step] (CF 上游 uniform_timestep=True).

    `scheduler.timesteps` 是 length-1000 tensor (1000 → 0); 取其中索引 [min_step, max_step]
    (这里 index 与 timestep 数值近似 1:1 因为 1000-step 离散).
    """
    timesteps = scheduler.timesteps.to(device)
    # min_step / max_step 是 0..1000 区间的 sigma·1000 值; 我们从 timesteps 里随机取一个
    valid = timesteps[(timesteps >= min_step) & (timesteps <= max_step)]
    idx = torch.randint(
        low=0, high=len(valid), size=(batch_size,), generator=generator, device=device,
    )
    return valid[idx]


def add_noise_per_frame(
    x0: torch.Tensor,
    noise: torch.Tensor,
    timestep_per_frame: torch.Tensor,
    scheduler: FlowMatchScheduler,
) -> torch.Tensor:
    """逐帧加噪: x_t[:, f] = (1-σ_f)·x0[:, f] + σ_f·noise[:, f].

    `x0`, `noise`: `[B, F, C, H, W]`
    `timestep_per_frame`: `[B, F]` (frame 0 anchor 时该位置为 0)
    """
    sigmas = scheduler.sigmas.to(x0.device)  # [num_train_timestep]
    timesteps = scheduler.timesteps.to(x0.device)  # [num_train_timestep]
    # 找最近的 timestep id (per element)
    # timestep_per_frame: [B, F] → flat [B*F] → 找索引 → reshape
    flat = timestep_per_frame.reshape(-1)
    diff = (timesteps[None, :] - flat[:, None]).abs()  # [B*F, T]
    ids = diff.argmin(dim=1)  # [B*F]
    sigma_per_frame = sigmas[ids].reshape(timestep_per_frame.shape)  # [B, F]
    sigma_per_frame = sigma_per_frame.to(x0.dtype)
    # broadcast to [B, F, 1, 1, 1]
    s = sigma_per_frame[:, :, None, None, None]
    return (1.0 - s) * x0 + s * noise


# --------------------------------------------------------------------- model_fn
def model_fn_causal_forcing(
    model: Any,
    *,
    tf_mode: bool,
    noisy_x: torch.Tensor,
    timestep: torch.Tensor,
    clean_x: torch.Tensor | None = None,
    kv_cache: dict | None = None,
    action: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    text_cache: dict | None = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """编排 CausalForcingReactiveGWMModel 两路径 forward.

    - `tf_mode=True, clean_x=x0`     : TF dual-block (Stage 1/2 训练)
        - `noisy_x`/`clean_x`: `[B, F, C, H, W]`
        - `timestep`:    `[B, F]` per-frame timestep (frame 0=0 if anchor)
        - 返回:           `[B, F, C, H, W]` noisy 半的 flow 预测 (clean 半丢弃)
    - `tf_mode=False, kv_cache=...`  : KV-cache 自回归 (sanity / Stage 3 / 推理)
        - `noisy_x`:     `[B, 1, C, H, W]` (单 latent frame)
        - `timestep`:    `[B, 1]`
        - `kv_cache`:    持续 mutate 的 cache dict
        - 返回:           `[B, 1, C, H, W]` 当前帧 flow 预测
    """
    return model.forward(
        noisy_x=noisy_x,
        timestep=timestep,
        clean_x=clean_x if tf_mode else None,
        kv_cache=kv_cache if not tf_mode else None,
        action=action,
        context=context,
        text_cache=text_cache,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        **kwargs,
    )


# --------------------------------------------------------------------- multi-step sampling (sanity)
@torch.no_grad()
def sample_multistep_kv_cache(
    model: Any,
    *,
    action_per_frame: torch.Tensor,
    context: torch.Tensor,
    first_frame_latent: torch.Tensor,
    target_latent_frames: int,
    diffusion_steps: int,
    kv_window_size: int,
    sink_size: int,
    num_frame_per_block: int = 1,
    scheduler_shift: float = 5.0,
    context_neg: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """KV-cache 自回归长视频生成 (CF 上游 `pipeline/causal_diffusion_inference.py` 风格).

    1. 初始化 KV-cache (+ 若 CFG 再开一份 neg cache) + text cache.
    2. refill_kv_cache(first_frame_latent, t=0): 首帧 anchor 写入 (cache-only; CFG 时 pos/neg 各一次).
    3. 逐 latent 帧 i = 1 .. target_latent_frames:
         a) noisy_x = randn [B, 1, C, H, W]
         b) 取 diffusion_steps 个 timestep (从 t_max → t_min, Wan flow-match):
              forward(pos) → flow_cond; 若 CFG 再 forward(neg) → flow_uncond;
              flow = flow_uncond + g·(flow_cond − flow_uncond) → scheduler.step → noisy_x_next
         c) 去噪完后得 x0_i; refill_kv_cache(x0_i, t=0) (CFG 时 pos/neg 各一次).
    4. 返回 `[B, target_latent_frames, C, H, W]` 视频 latent.

    CFG (对齐 CF 上游 `causal_diffusion_inference.py:206-226`): 给 `context_neg` 且
    `guidance_scale != 1.0` 时启用——pos/neg 各维护独立 KV-cache (cond/uncond forward 写不同
    K/V), 每步两次 forward 后做 `flow_uncond + g·(flow_cond − flow_uncond)`; neg 分支用同一份
    action, 只换 text context. `context_neg=None` (默认) → 单路正 prompt, 无 CFG (verify 用).

    `action_per_frame`: `[B, T_pixel, num_buttons]`; 内部按帧切分供 model.forward 用.
    `first_frame_latent`: `[B, 1, C, H, W]` (anchor latent, 不加噪).
    """
    if device is None:
        device = first_frame_latent.device

    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(
        num_inference_steps=diffusion_steps,
        denoising_strength=1.0,
        training=False,
        shift=scheduler_shift,
    )
    sigmas = sched.sigmas.to(device)
    timesteps = sched.timesteps.to(device)

    B, _, C, H, W = first_frame_latent.shape

    # Discover frame_seqlen by running patchify on the first frame (no grad).
    with torch.no_grad():
        _, _, h_lat, w_lat = model._patchify_and_geom(first_frame_latent)
    frame_seqlen = h_lat * w_lat

    use_cfg = context_neg is not None and guidance_scale != 1.0

    # KV-cache init (CFG 时 pos/neg 各一份: cond/uncond forward 写不同 K/V, 不能共享 cache)
    kv_cache = model.init_kv_cache(
        batch_size=B, device=device, dtype=dtype,
        frame_seqlen=frame_seqlen,
        kv_window_size=kv_window_size, sink_size=sink_size,
    )
    kv_cache_neg = (
        model.init_kv_cache(
            batch_size=B, device=device, dtype=dtype,
            frame_seqlen=frame_seqlen,
            kv_window_size=kv_window_size, sink_size=sink_size,
        )
        if use_cfg else None
    )

    # 1) refill first frame (anchor, t=0) — writes K/V into cache for all subsequent frames.
    #    action bin 对齐 teacher/critic 的 adaptive-pool, 覆盖全部 pixel 帧.
    model.refill_kv_cache(
        x_block=first_frame_latent,
        action=_action_bin_for_frame(action_per_frame, 0, target_latent_frames),
        context=context,
        kv_cache=kv_cache,
    )
    if use_cfg:
        model.refill_kv_cache(
            x_block=first_frame_latent,
            action=_action_bin_for_frame(action_per_frame, 0, target_latent_frames),
            context=context_neg,
            kv_cache=kv_cache_neg,
        )

    output_frames = [first_frame_latent]

    # 2) Roll out frame-by-frame
    for i in range(1, target_latent_frames):
        noisy_x = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)
        action_i = _action_bin_for_frame(action_per_frame, i, target_latent_frames)
        for step_id, t in enumerate(timesteps):
            t_curr = t.unsqueeze(0).expand(B, 1).to(dtype)
            # write_to_cache=False so each diffusion step's K/V are scratch —
            # do NOT advance global_end_index. Otherwise the RoPE start_frame
            # blows past the precomputed 1024-entry table within ~21 frames.
            flow_cond = model.forward(
                noisy_x=noisy_x,
                timestep=t_curr,
                clean_x=None,
                kv_cache=kv_cache,
                action=action_i,
                context=context,
                write_to_cache=False,
            )
            if use_cfg:
                # uncond 分支: 同一 action, 只换 text context → neg cache (CF 上游 CFG).
                flow_uncond = model.forward(
                    noisy_x=noisy_x,
                    timestep=t_curr,
                    clean_x=None,
                    kv_cache=kv_cache_neg,
                    action=action_i,
                    context=context_neg,
                    write_to_cache=False,
                )
                flow_pred = flow_uncond + guidance_scale * (flow_cond - flow_uncond)
            else:
                flow_pred = flow_cond
            sigma_curr = sigmas[step_id]
            sigma_next = (
                sigmas[step_id + 1]
                if step_id + 1 < len(sigmas)
                else torch.zeros((), device=device, dtype=sigma_curr.dtype)
            )
            noisy_x = noisy_x + flow_pred * (sigma_next - sigma_curr)

        x0_i = noisy_x  # final clean latent for this frame
        # 3) refill writes K/V using the clean latent (CF 上游 Step 3.3); CFG 时 pos/neg 各一次.
        model.refill_kv_cache(
            x_block=x0_i,
            action=action_i,
            context=context,
            kv_cache=kv_cache,
        )
        if use_cfg:
            model.refill_kv_cache(
                x_block=x0_i,
                action=action_i,
                context=context_neg,
                kv_cache=kv_cache_neg,
            )
        output_frames.append(x0_i)

    return torch.cat(output_frames, dim=1)


@torch.no_grad()
def sample_dmd_kv_cache(
    model: Any,
    *,
    action_per_frame: torch.Tensor,
    context: torch.Tensor,
    first_frame_latent: torch.Tensor,
    target_latent_frames: int,
    diffusion_steps: int,
    kv_window_size: int,
    sink_size: int,
    num_frame_per_block: int = 1,
    scheduler_shift: float = 5.0,
    context_neg: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Stage 3 DMD/self-forcing KV-cache rollout with re-noise sampler.

    This mirrors the training-time DMD2 backward simulation used by
    ``dmd_self_rollout`` / ``dmd_long_tail_self_rollout``: each intermediate
    step predicts ``x0 = x_t - sigma_t * flow`` and re-noises that x0 to the
    next sigma with fresh Gaussian noise. This differs from
    ``sample_multistep_kv_cache``, which integrates the flow ODE directly.
    """
    if device is None:
        device = first_frame_latent.device

    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(
        num_inference_steps=diffusion_steps,
        denoising_strength=1.0,
        training=False,
        shift=scheduler_shift,
    )
    sigmas = sched.sigmas.to(device)
    timesteps = sched.timesteps.to(device)

    B, _, C, H, W = first_frame_latent.shape

    with torch.no_grad():
        _, _, h_lat, w_lat = model._patchify_and_geom(first_frame_latent)
    frame_seqlen = h_lat * w_lat

    use_cfg = context_neg is not None and guidance_scale != 1.0

    kv_cache = model.init_kv_cache(
        batch_size=B, device=device, dtype=dtype,
        frame_seqlen=frame_seqlen,
        kv_window_size=kv_window_size, sink_size=sink_size,
    )
    kv_cache_neg = (
        model.init_kv_cache(
            batch_size=B, device=device, dtype=dtype,
            frame_seqlen=frame_seqlen,
            kv_window_size=kv_window_size, sink_size=sink_size,
        )
        if use_cfg else None
    )

    model.refill_kv_cache(
        x_block=first_frame_latent,
        action=_action_bin_for_frame(action_per_frame, 0, target_latent_frames),
        context=context,
        kv_cache=kv_cache,
    )
    if use_cfg:
        model.refill_kv_cache(
            x_block=first_frame_latent,
            action=_action_bin_for_frame(action_per_frame, 0, target_latent_frames),
            context=context_neg,
            kv_cache=kv_cache_neg,
        )

    output_frames = [first_frame_latent]

    for i in range(1, target_latent_frames):
        noisy_x = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)
        action_i = _action_bin_for_frame(action_per_frame, i, target_latent_frames)
        x0_i: torch.Tensor | None = None
        for step_id, t in enumerate(timesteps):
            t_curr = t.unsqueeze(0).expand(B, 1).to(dtype)
            flow_cond = model.forward(
                noisy_x=noisy_x,
                timestep=t_curr,
                clean_x=None,
                kv_cache=kv_cache,
                action=action_i,
                context=context,
                write_to_cache=False,
            )
            if use_cfg:
                flow_uncond = model.forward(
                    noisy_x=noisy_x,
                    timestep=t_curr,
                    clean_x=None,
                    kv_cache=kv_cache_neg,
                    action=action_i,
                    context=context_neg,
                    write_to_cache=False,
                )
                flow_pred = flow_uncond + guidance_scale * (flow_cond - flow_uncond)
            else:
                flow_pred = flow_cond

            sigma_curr = sigmas[step_id].to(dtype)
            x0_i = noisy_x - sigma_curr * flow_pred
            if step_id + 1 < len(sigmas):
                sigma_next = sigmas[step_id + 1].to(dtype)
                noisy_x = (1.0 - sigma_next) * x0_i + sigma_next * torch.randn_like(x0_i)
            else:
                noisy_x = x0_i

        assert x0_i is not None
        model.refill_kv_cache(
            x_block=x0_i,
            action=action_i,
            context=context,
            kv_cache=kv_cache,
        )
        if use_cfg:
            model.refill_kv_cache(
                x_block=x0_i,
                action=action_i,
                context=context_neg,
                kv_cache=kv_cache_neg,
            )
        output_frames.append(x0_i)

    return torch.cat(output_frames, dim=1)



def _action_slice_len(action_per_frame: torch.Tensor, latent_block: int, total_latent: int) -> int:
    """每 latent 帧对应的 pixel-frame action 切片长度 (整数下取). [DEPRECATED: 用 _action_bin_for_frame]

    floor(T/F) 会丢尾部 (101//26=3 → 26*3=78<101 丢 23 帧), 且与 teacher/critic 的
    adaptive_max_pool1d 分桶不一致。保留仅为兼容,不再用于 rollout / sanity。
    """
    T_pixel = action_per_frame.shape[1]
    return max(1, T_pixel // total_latent) * latent_block


def _action_bin_for_frame(action_per_frame: torch.Tensor, frame_idx: int, total_latent: int) -> torch.Tensor:
    """latent 帧 ``frame_idx`` 的 action 切片,对齐 ``adaptive_max_pool1d(T → total_latent)`` 分桶。

    PyTorch adaptive pooling 把输出 bin i 映射到输入区间 ``[floor(i*T/F), ceil((i+1)*T/F))``。
    rollout / sanity 用**同一**区间 → 因果 generator 的逐帧 action 与 teacher/critic 的
    完整-action adaptive-pool 打分 (``prepare_action_binned``) 一致,且覆盖全部 T 个 pixel 帧、
    不丢尾部 (修 101//26=3 丢 23 帧的问题)。
    """
    T = int(action_per_frame.shape[1])
    F = max(1, int(total_latent))
    start = (frame_idx * T) // F
    end = -((-((frame_idx + 1) * T)) // F)  # ceil((frame_idx+1)*T/F)
    end = min(max(end, start + 1), T)
    return action_per_frame[:, start:end]


# ===================================================================== Stage 2 CD helpers
# 追加 (Stage 2); 不改上方任何 Stage 1 函数 / model_fn 默认返回类型 (保护规则 1/2).
def build_cd_scheduler(num_nodes: int = 48, sigma_shift: float = 5.0) -> FlowMatchScheduler:
    """CD 离散调度: ``FlowMatchScheduler("Wan").set_timesteps(N)``.

    ``set_timesteps_wan`` 内部 ``linspace(1, 0, N+1)[:-1]`` → 恰好 **N 个节点**, 天然丢掉
    σ=0 端点 (避开 t=0 退化 / 1/σ 奇异), 与 CF 上游 ``causal_cd_framewise.yaml`` 的
    extra_one_step 行为一致. N=48 → index 0 (σ≈1, 高噪) .. 47 (小 σ>0, 低噪).
    """
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(
        num_inference_steps=num_nodes,
        denoising_strength=1.0,
        training=False,
        shift=sigma_shift,
    )
    return sched


def flow_to_x0(
    latent: torch.Tensor, flow: torch.Tensor, sigma_per_frame: torch.Tensor
) -> torch.Tensor:
    """flow 预测 → x0 估计: ``x0 = latent − σ·flow``.

    Wan flow-match: ``x_σ = (1−σ)·x0 + σ·ε``, flow target = ``ε − x0 = dx/dσ``
    ⇒ ``x0 = x_σ − σ·(ε−x0) = latent − σ·flow``.
    ``sigma_per_frame``: ``[B, F]`` (frame 0 anchor → σ=0 ⇒ ``x0[:,0]=latent[:,0]``).

    NOTE: ``model_fn_causal_forcing`` / ``model.forward`` 默认仍只返回 flow; 此 helper 是
    Stage 2 CD 在外部做的转换, **不改默认返回类型** (保护规则 1/2).
    """
    s = sigma_per_frame[:, :, None, None, None].to(latent.dtype)
    return latent - s * flow


@torch.no_grad()
def cd_teacher_one_step(
    teacher: Any,
    *,
    latent_t: torch.Tensor,
    timestep_per_frame: torch.Tensor,
    t_idx: int,
    scheduler: FlowMatchScheduler,
    clean_x: torch.Tensor,
    action: torch.Tensor | None,
    context_pos: torch.Tensor,
    context_neg: torch.Tensor,
    guidance_scale: float,
    first_frame_x0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Teacher 单步 ODE 推进 (节点 ``t_idx`` → ``t_idx+1``) + negative-prompt CFG.

    - teacher 走 TF dual-block 路径 (``clean_x`` 上下文), forward 两次 (pos / neg context);
    - CFG flow: ``v = v_uncond + g·(v_cond − v_uncond)``;
    - 单步 flow-match ODE: ``latent_t_next = latent_t − (t − t_next)/1000 · v``
      ( = ``latent_t + (σ_next − σ_curr)·v``, 与 ``sample_multistep_kv_cache`` 同号);
    - frame 0 anchor (I2V): 若给 ``first_frame_x0``, 推进后强制
      ``latent_t_next[:,0]=first_frame_x0[:,0]``.

    全程 ``no_grad`` (teacher 冻结). ``timestep_per_frame`` 为 uniform t (frame0=0) 的
    ``[B, F]`` broadcast 张量.
    """
    timesteps = scheduler.timesteps.to(latent_t.device)
    t_curr = timesteps[t_idx]
    t_next = (
        timesteps[t_idx + 1] if t_idx + 1 < len(timesteps) else torch.zeros_like(t_curr)
    )
    v_cond = model_fn_causal_forcing(
        teacher, tf_mode=True, noisy_x=latent_t, timestep=timestep_per_frame,
        clean_x=clean_x, action=action, context=context_pos,
    )
    v_uncond = model_fn_causal_forcing(
        teacher, tf_mode=True, noisy_x=latent_t, timestep=timestep_per_frame,
        clean_x=clean_x, action=action, context=context_neg,
    )
    v = v_uncond + guidance_scale * (v_cond - v_uncond)
    dt = ((t_curr - t_next) / 1000.0).to(latent_t.dtype)
    latent_t_next = latent_t - dt * v
    if first_frame_x0 is not None:
        latent_t_next = latent_t_next.clone()
        latent_t_next[:, 0] = first_frame_x0[:, 0]
    return latent_t_next


# ===================================================================== Stage 3 DMD helpers
# 追加 (Stage 3); 不改上方任何 Stage 1/2 函数 / model_fn 默认返回 flow (保护规则 1/2).
# 上游真实文件: model/dmd.py + model/base.py + pipeline/self_forcing_training.py
#               + trainer/distillation.py (见 stage3_cf_alignment_review.md §1).
# 角色: generator=CausalForcingReactiveGWMModel (因果, KV-cache rollout);
#       teacher/critic=双向 ReactiveGWMModel (父类, model_fn_wan_video 打分).
# 关键: Wan flow-match 下 σ = t/1000 精确成立 → flow↔x0 直接算, 不用 argmin.


def warp_denoising_steps(
    scheduler1000: FlowMatchScheduler,
    denoising_step_list: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """上游 warp (model/base.py:22-24): ``timesteps[1000 − denoising_step_list]``.

    把 ``[1000,750,500,250]`` 映射到 1000-step 离散调度上的实际 (timestep, sigma).
    与 ``set_timesteps(4)`` 逐点等价 (已验证), 但此处直接对齐上游 warp 逻辑.
    返回 ``(warped_timesteps[K], warped_sigmas[K])``; sigma = timestep/1000.
    """
    step_list = torch.tensor(list(denoising_step_list), dtype=torch.long, device=device)
    # append a 0 (= σ=0 端点) 以兼容 step=0 的情形 (本项目 [1000..250] 用不到, 但对齐上游).
    ts = torch.cat([scheduler1000.timesteps.to(device), torch.zeros(1, device=device)])
    sg = torch.cat([scheduler1000.sigmas.to(device), torch.zeros(1, device=device)])
    idx = 1000 - step_list
    return ts[idx], sg[idx]


def sample_dmd_timestep(
    num_train_timestep: int,
    timestep_shift: float,
    clamp_min: float,
    clamp_max: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DMD score / critic timestep: uniform ``[0, num_train_timestep)`` → shift → clamp.

    对齐上游 ``model/dmd.py:158-172,283-287`` (``ts_schedule=false`` 分支:
    ``min=min_score_timestep=0``, ``max=num_train_timestep=1000``). 返回标量 ``[1]`` (float32).
    """
    t = torch.randint(0, int(num_train_timestep), (1,), device=device, generator=generator).float()
    if timestep_shift and timestep_shift > 1:
        tt = t / 1000.0
        t = timestep_shift * tt / (1.0 + (timestep_shift - 1.0) * tt) * 1000.0
    return t.clamp(float(clamp_min), float(clamp_max))


def _sample_shared_exit_index(num_steps: int, device: torch.device) -> int:
    """每次 rollout 采一个随机 exit step; ``same_step_across_blocks=true`` → 所有帧共享.

    DDP: 必须跨 rank 同步 (否则各 rank 的计算图结构 = 哪一步留梯度 不同 → DDP allreduce
    mismatch / hang). 对齐上游 ``pipeline/self_forcing_training.py::generate_and_sync_list``.
    """
    idx = torch.randint(0, int(num_steps), (1,), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(idx, src=0)
    return int(idx.item())


@torch.no_grad()
def _refill_anchor(generator, first_frame_latent, action_slice, context, kv_cache) -> None:
    generator.refill_kv_cache(
        x_block=first_frame_latent, action=action_slice, context=context, kv_cache=kv_cache,
    )


def _snapshot_kv_cache_read_state(kv_cache: dict) -> dict:
    """Shallow snapshot for checkpoint recompute.

    ``write_to_cache=False`` forwards only need historical K/V buffers plus current
    local/global end indices. Buffers are append-only for the 26-frame training window,
    so sharing K/V storage is safe; cloning the scalar indices prevents checkpoint
    recompute from seeing a later, longer cache.
    """
    return {
        "frame_seqlen": kv_cache["frame_seqlen"],
        "self": [
            {
                **layer,
                "global_end_index": layer["global_end_index"].clone(),
                "local_end_index": layer["local_end_index"].clone(),
            }
            for layer in kv_cache["self"]
        ],
        "cross": [
            {"k": layer["k"], "v": layer["v"], "is_init": layer["is_init"]}
            for layer in kv_cache["cross"]
        ],
    }


def dmd_self_rollout(
    generator: Any,
    *,
    first_frame_latent: torch.Tensor,   # [B, 1, C, H, W] GT anchor (detached)
    action_per_frame: torch.Tensor,     # [B, T_pixel, num_buttons]
    context: torch.Tensor,              # T5 [B, L, dim] (positive prompt)
    num_frames: int,                    # 26 (frame0 anchor + 25 generated)
    warped_timesteps: torch.Tensor,     # [K] warped 4-step timesteps
    warped_sigmas: torch.Tensor,        # [K] = warped_timesteps / 1000
    kv_window_size: int,                # recent window size for training rollout
    sink_size: int,                     # number of persistent sink frames
    keep_grad: bool,                    # True: generator phase; False: critic phase
    last_step_only: bool = False,
    use_gradient_checkpointing: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, int]:
    """DMD2 backward-simulation KV-cache self-rollout (上游 ``self_forcing_training.py``).

    与 sanity 推理的确定性 ODE step (``sample_multistep_kv_cache``) **不同**: 这里每个
    去噪步去到 x0 后用**全新随机噪声**重加噪到下一步 (一致性采样器 / DMD2 Sec 4.5).

    - 26 帧定长: frame0 = 数据集 GT anchor (先 refill 进 cache, 不生成); frame 1..25 rollout.
    - shared random ``exit_idx`` (``same_step_across_blocks=true``): 整段 rollout 共享.
    - 非 exit 步 + 每帧完成后的 refill 走 ``no_grad``; 仅 ``keep_grad`` 时的 exit-step forward
      进计算图 (历史 K/V 是 refill 写入的 detached 值 → 25 帧 exit forward 在图里相互独立).
    - 训练 rollout 使用 sink+recent buffer：保留 ``sink_size`` 个开头帧 + 最近 ``kv_window_size`` 个历史帧。

    返回 ``(video [B, num_frames, C, H, W], exit_idx)``; frame0 = anchor (无梯度),
    frame 1..25 在 ``keep_grad`` 下带梯度.

    ``use_gradient_checkpointing=True`` 时, exit-step 的 ``write_to_cache=False`` forward
    会 checkpoint 重算以压低 26 帧整窗 backward 激活峰值；不改变 rollout 数学语义。
    """
    if device is None:
        device = first_frame_latent.device
    B, _, C, H, W = first_frame_latent.shape
    num_steps = int(warped_timesteps.shape[0])

    with torch.no_grad():
        _, _, h_lat, w_lat = generator._patchify_and_geom(first_frame_latent)
    frame_seqlen = h_lat * w_lat

    kv_cache = generator.init_kv_cache(
        batch_size=B, device=device, dtype=dtype, frame_seqlen=frame_seqlen,
        kv_window_size=kv_window_size, sink_size=sink_size,
    )
    exit_idx = (num_steps - 1) if last_step_only else _sample_shared_exit_index(num_steps, device)

    # frame0 anchor → cache (cache-only refill, no_grad). action bin 对齐 score 的 adaptive-pool.
    _refill_anchor(generator, first_frame_latent, _action_bin_for_frame(action_per_frame, 0, num_frames), context, kv_cache)
    output_frames = [first_frame_latent]

    for i in range(1, num_frames):
        action_i = _action_bin_for_frame(action_per_frame, i, num_frames)
        noisy = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)
        denoised: torch.Tensor | None = None
        for index in range(num_steps):
            t_pf = warped_timesteps[index].to(dtype).view(1, 1).expand(B, 1)
            sigma = warped_sigmas[index].to(dtype)
            is_exit = index == exit_idx
            grad_on = is_exit and keep_grad
            ctx = torch.enable_grad() if grad_on else torch.no_grad()
            with ctx:
                kv_cache_read = (
                    _snapshot_kv_cache_read_state(kv_cache)
                    if grad_on and use_gradient_checkpointing else kv_cache
                )

                def _forward(noisy_arg, *, _t_pf=t_pf, _action_i=action_i,
                             _context=context, _kv_cache_read=kv_cache_read):
                    return generator.forward(
                        noisy_x=noisy_arg, timestep=_t_pf, clean_x=None, kv_cache=_kv_cache_read,
                        action=_action_i, context=_context, write_to_cache=False,
                    )
                if grad_on and use_gradient_checkpointing:
                    flow = torch.utils.checkpoint.checkpoint(
                        _forward, noisy, use_reentrant=False,
                    )
                else:
                    flow = _forward(noisy)
                x0 = noisy - sigma * flow
            if is_exit:
                denoised = x0
                break
            # consistency sampler: re-noise x0 to next step with FRESH noise (≠ ODE step)
            sigma_next = warped_sigmas[index + 1].to(dtype)
            noisy = (1.0 - sigma_next) * x0 + sigma_next * torch.randn_like(x0)
        # commit clean frame to history (detached → 后续帧 attend 到 detached K/V)
        _refill_anchor(generator, denoised.detach(), action_i, context, kv_cache)
        output_frames.append(denoised)

    return torch.cat(output_frames, dim=1), exit_idx


def dmd_slice_tail_action(
    action_per_frame: torch.Tensor | None,
    tail_start: int,
    total_latent: int,
) -> torch.Tensor | None:
    """Slice raw pixel-frame actions for the tail DMD window.

    The generator rolls out with the full action sequence for ``total_latent`` frames,
    while teacher/critic score only the tail window. Passing the full action here would
    adaptive-pool the wrong time span, so use the raw suffix aligned to ``tail_start``.
    """
    if action_per_frame is None:
        return None
    T = int(action_per_frame.shape[1])
    F = max(1, int(total_latent))
    start = (int(tail_start) * T) // F
    start = min(max(start, 0), T - 1)
    return action_per_frame[:, start:].contiguous()


def dmd_long_tail_self_rollout(
    generator: Any,
    *,
    first_frame_latent: torch.Tensor,
    action_per_frame: torch.Tensor,
    context: torch.Tensor,
    num_frames: int,
    score_frames: int,
    warped_timesteps: torch.Tensor,
    warped_sigmas: torch.Tensor,
    kv_window_size: int,
    sink_size: int,
    keep_grad: bool,
    last_step_only: bool = False,
    use_gradient_checkpointing: bool = False,
    reanchor_decode_latents: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Streaming long rollout, returning only the tail DMD window.

    ``num_frames`` is the full generator horizon ``L``. ``score_frames`` is the tail
    DMD window length ``S``. Prefix frames up to ``tail_start = L-S`` are generated
    under ``no_grad`` and only advance the KV cache. For generator phase, only frames
    ``tail_start+1 .. L-1`` keep the exit-step graph, so memory stays bounded by the
    short score window rather than the full rollout horizon.

    Returns ``(tail_video, reanchor_latents, exit_idx)`` where ``tail_video`` is
    ``[anchor_candidate, tail generated frames...]`` with shape ``[B,S,C,H,W]`` and
    ``reanchor_latents`` is the short latent chunk ending at the anchor candidate.
    """
    if device is None:
        device = first_frame_latent.device
    num_frames = int(num_frames)
    score_frames = int(score_frames)
    if score_frames < 1 or num_frames < score_frames:
        raise ValueError(f"invalid long-tail rollout: L={num_frames}, S={score_frames}")

    B, _, C, H, W = first_frame_latent.shape
    num_steps = int(warped_timesteps.shape[0])
    tail_start = num_frames - score_frames
    reanchor_decode_latents = max(1, int(reanchor_decode_latents))

    with torch.no_grad():
        _, _, h_lat, w_lat = generator._patchify_and_geom(first_frame_latent)
    frame_seqlen = h_lat * w_lat

    kv_cache = generator.init_kv_cache(
        batch_size=B, device=device, dtype=dtype, frame_seqlen=frame_seqlen,
        kv_window_size=kv_window_size, sink_size=sink_size,
    )
    exit_idx = (num_steps - 1) if last_step_only else _sample_shared_exit_index(num_steps, device)

    _refill_anchor(
        generator,
        first_frame_latent,
        _action_bin_for_frame(action_per_frame, 0, num_frames),
        context,
        kv_cache,
    )

    anchor_candidate = first_frame_latent.detach()
    reanchor_history: list[torch.Tensor] = [first_frame_latent.detach()]
    reanchor_latents = first_frame_latent.detach()
    tail_frames: list[torch.Tensor] = []

    for i in range(1, num_frames):
        action_i = _action_bin_for_frame(action_per_frame, i, num_frames)
        noisy = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)
        denoised: torch.Tensor | None = None
        frame_keep_grad = keep_grad and i > tail_start
        for index in range(num_steps):
            t_pf = warped_timesteps[index].to(dtype).view(1, 1).expand(B, 1)
            sigma = warped_sigmas[index].to(dtype)
            is_exit = index == exit_idx
            grad_on = is_exit and frame_keep_grad
            ctx = torch.enable_grad() if grad_on else torch.no_grad()
            with ctx:
                kv_cache_read = (
                    _snapshot_kv_cache_read_state(kv_cache)
                    if grad_on and use_gradient_checkpointing else kv_cache
                )

                def _forward(noisy_arg, *, _t_pf=t_pf, _action_i=action_i,
                             _context=context, _kv_cache_read=kv_cache_read):
                    return generator.forward(
                        noisy_x=noisy_arg, timestep=_t_pf, clean_x=None, kv_cache=_kv_cache_read,
                        action=_action_i, context=_context, write_to_cache=False,
                    )
                if grad_on and use_gradient_checkpointing:
                    flow = torch.utils.checkpoint.checkpoint(
                        _forward, noisy, use_reentrant=False,
                    )
                else:
                    flow = _forward(noisy)
                x0 = noisy - sigma * flow
            if is_exit:
                denoised = x0
                break
            sigma_next = warped_sigmas[index + 1].to(dtype)
            noisy = (1.0 - sigma_next) * x0 + sigma_next * torch.randn_like(x0)

        assert denoised is not None
        _refill_anchor(generator, denoised.detach(), action_i, context, kv_cache)

        if i <= tail_start:
            anchor_candidate = denoised.detach()
            reanchor_history.append(anchor_candidate)
            if len(reanchor_history) > reanchor_decode_latents:
                reanchor_history.pop(0)
            if i == tail_start:
                reanchor_latents = torch.cat(reanchor_history, dim=1).detach().contiguous()
        else:
            tail_frames.append(denoised)

    if tail_start == 0:
        reanchor_latents = torch.cat(reanchor_history[-reanchor_decode_latents:], dim=1).detach().contiguous()

    tail_video = torch.cat([anchor_candidate] + tail_frames, dim=1)
    return tail_video, reanchor_latents, exit_idx


def _dmd_score_flow(
    score_model: Any,
    noisy_bfchw: torch.Tensor,
    t_scalar: torch.Tensor,
    context: torch.Tensor,
    action: torch.Tensor | None,
    *,
    use_gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """双向 teacher/critic 打分: ``model_fn_wan_video`` → flow ``[B, F, C, H, W]``.

    ``fuse_vae_embedding_in_latents=True`` → frame0 自动 timestep=0 (I2V anchor).
    timestep cast 到 latent dtype (与 base pipeline 用法一致, 避免 cat dtype mismatch).
    """
    latents = noisy_bfchw.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W]
    flow = model_fn_wan_video(
        dit=score_model,
        latents=latents,
        timestep=t_scalar.to(latents.dtype),
        context=context,
        keyboard_action=action,
        fuse_vae_embedding_in_latents=True,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return flow.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W]


def dmd_score_x0(
    score_model: Any,
    noisy_bfchw: torch.Tensor,
    t_scalar: torch.Tensor,
    context: torch.Tensor,
    action: torch.Tensor | None,
    sigma_t: float,
    *,
    use_gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """teacher/critic x0 预测: ``x0 = noisy − σ_pf·flow`` (per-frame σ, frame0=0).

    frame0 (I2V anchor) σ=0 ⇒ ``x0[:,0]=noisy[:,0]=clean``; model_fn 也把 frame0 当 t=0.
    """
    flow = _dmd_score_flow(
        score_model, noisy_bfchw, t_scalar, context, action,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    B, Fn = noisy_bfchw.shape[:2]
    sigma_pf = torch.full((B, Fn), float(sigma_t), device=noisy_bfchw.device, dtype=noisy_bfchw.dtype)
    sigma_pf[:, 0] = 0.0
    return noisy_bfchw - sigma_pf[:, :, None, None, None] * flow


def dmd_distribution_matching_loss(
    pred_video_bfchw: torch.Tensor,
    *,
    teacher: Any,
    critic: Any,
    context_pos: torch.Tensor,
    context_neg: torch.Tensor,
    action: torch.Tensor | None,
    real_guidance_scale: float,
    fake_guidance_scale: float,
    num_train_timestep: int,
    timestep_shift: float,
    clamp_min: float,
    clamp_max: float,
    first_frame_anchor: bool = True,
) -> torch.Tensor:
    """DMD generator loss (x0 空间; 上游 ``model/dmd.py:56-197``).

    teacher CFG ``real_x0 = cond + g·(cond−uncond)`` (g=real_guidance_scale);
    critic cond-only (fake_guidance_scale=0); ``grad = fake_x0 − real_x0``;
    ``normalizer = mean(abs(pred − real_x0))[1,2,3,4] keepdim`` (无 clamp) → ``nan_to_num``;
    ``loss = 0.5·MSE(pred.double(), (pred − grad).detach().double())``.

    score 全程 ``no_grad``; 只有 ``pred_video_bfchw`` (rollout 产物) 带 grad 进 loss.
    frame0 (I2V anchor): 只给 frame 1..25 加噪、frame0 保持 clean; frame0 在 grad/normalizer
    贡献恒为 0, 且排除出 loss (项目适配, 见 stage3_cf_alignment_review.md §4.5 + implement 自决项 1).
    """
    B, Fn = pred_video_bfchw.shape[:2]
    dev = pred_video_bfchw.device
    anchor = first_frame_anchor and Fn > 1
    with torch.no_grad():
        t = sample_dmd_timestep(num_train_timestep, timestep_shift, clamp_min, clamp_max, dev)
        sigma_t = float((t / 1000.0).item())
        noise = torch.randn_like(pred_video_bfchw)
        noisy = pred_video_bfchw.clone()
        if anchor:
            noisy[:, 1:] = (1.0 - sigma_t) * pred_video_bfchw[:, 1:] + sigma_t * noise[:, 1:]
        else:
            noisy = (1.0 - sigma_t) * pred_video_bfchw + sigma_t * noise

        fake_x0 = dmd_score_x0(critic, noisy, t, context_pos, action, sigma_t)
        if fake_guidance_scale != 0.0:
            fake_x0_u = dmd_score_x0(critic, noisy, t, context_neg, action, sigma_t)
            fake_x0 = fake_x0 + (fake_x0 - fake_x0_u) * fake_guidance_scale

        real_x0_c = dmd_score_x0(teacher, noisy, t, context_pos, action, sigma_t)
        real_x0_u = dmd_score_x0(teacher, noisy, t, context_neg, action, sigma_t)
        real_x0 = real_x0_c + (real_x0_c - real_x0_u) * real_guidance_scale

        grad = fake_x0 - real_x0
        if anchor:
            p_real = (pred_video_bfchw[:, 1:] - real_x0[:, 1:]).abs().mean(dim=[1, 2, 3, 4], keepdim=True)
        else:
            p_real = (pred_video_bfchw - real_x0).abs().mean(dim=[1, 2, 3, 4], keepdim=True)
        grad = torch.nan_to_num(grad / p_real)
        target = (pred_video_bfchw - grad).detach()

    if anchor:
        return 0.5 * F.mse_loss(pred_video_bfchw[:, 1:].double(), target[:, 1:].double())
    return 0.5 * F.mse_loss(pred_video_bfchw.double(), target.double())


def dmd_critic_flow_loss(
    generated_bfchw: torch.Tensor,
    *,
    critic: Any,
    context_pos: torch.Tensor,
    action: torch.Tensor | None,
    num_train_timestep: int,
    timestep_shift: float,
    clamp_min: float,
    clamp_max: float,
    first_frame_anchor: bool = True,
    use_gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """Critic flow-matching denoising loss (上游 ``model/dmd.py:240-335`` + ``utils/loss.py`` flow).

    在 generated (detach) 上加 uniform 噪声 (frame 1..25) → critic flow 预测 → flow MSE
    ``mean((critic_flow − (noise − generated))²)``. 这是上游 x0→flow→FlowPredLoss 在
    flow-native 模型下的等价式 (``flow_pred=(xt−x0)/σ`` 当 ``x0=xt−σ·flow`` ⇒ ``flow_pred=flow``).
    frame0 (I2V anchor) 排除 (未加噪, model_fn 当 t=0; 项目适配).
    """
    B, Fn = generated_bfchw.shape[:2]
    dev = generated_bfchw.device
    anchor = first_frame_anchor and Fn > 1
    t = sample_dmd_timestep(num_train_timestep, timestep_shift, clamp_min, clamp_max, dev)
    sigma_t = float((t / 1000.0).item())
    noise = torch.randn_like(generated_bfchw)
    noisy = generated_bfchw.clone()
    if anchor:
        noisy[:, 1:] = (1.0 - sigma_t) * generated_bfchw[:, 1:] + sigma_t * noise[:, 1:]
    else:
        noisy = (1.0 - sigma_t) * generated_bfchw + sigma_t * noise

    critic_flow = _dmd_score_flow(
        critic, noisy, t, context_pos, action,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    target_flow = noise - generated_bfchw  # true flow target = ε − x0
    if anchor:
        return F.mse_loss(critic_flow[:, 1:].float(), target_flow[:, 1:].float())
    return F.mse_loss(critic_flow.float(), target_flow.float())
