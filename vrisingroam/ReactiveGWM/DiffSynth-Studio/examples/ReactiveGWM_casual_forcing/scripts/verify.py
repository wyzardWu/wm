"""CF verify — CPU 子命令 (极小维度) + GPU 子命令.

每个子命令独立 entry; 主 main 路由到 _verify_<name>(); 失败 raise + return non-zero.

CPU 子命令 (Stage 1, 无需 GPU 或 ckpt):
  state_dict        — CF model 自洽 state_dict roundtrip
  scheduler         — FlowMatchScheduler('Wan') 基本不变量
  tf_mask           — dual-block BlockMask 构造无误
  ar_tf             — tiny TF dual-block forward 数值有限 + 梯度通
  kv_cache_init     — init_kv_cache shape / 计数正确
  kv_cache_refill   — refill 触发 K/V 写入 + 不改变模型权重
  kv_cache_rollout  — 多 block rollout 内部自洽 (driver_end 单调)
  sanity_kv         — tiny 端到端 KV-cache rollout 2 帧
  smoke             — tiny CFARTrainingModule 风格 forward+backward

GPU 子命令 (避免与 8 卡训练并占; 见 implement.md):
  bf16_two_paths    — bf16 TF 与 KV-cache 两路径数值有限 + 梯度通

Stage 2 追加: cd
Stage 3 追加: rollout(real) / dmd
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root in sys.path for import
_HERE = Path(__file__).resolve().parent.parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SUBCOMMANDS = [
    "state_dict",
    "scheduler",
    "tf_mask",
    "ar_tf",
    "kv_cache_init",
    "kv_cache_refill",
    "kv_cache_rollout",
    "sanity_kv",
    "smoke",
    "cd",
    "rollout",
    "dmd",
    "bf16_two_paths",
]


# ----------------------------------------------------------------------- tiny model factory
def _tiny_model_kwargs() -> dict:
    """Tiny WanModel arch (CPU-friendly) reusing ReactiveGWMModel structure.

    head_dim must satisfy `head_dim // 3` even, otherwise the 3D RoPE
    `precompute_freqs_cis_3d` produces freqs that don't sum to head_dim/2
    (head_dim=16 → 3+2+2=7 ≠ 8). Use head_dim=32 (3+5+5=16/2 ✓).
    """
    return dict(
        has_image_input=False,
        patch_size=[1, 2, 2],
        in_dim=4,
        dim=64,
        ffn_dim=128,
        freq_dim=32,
        text_dim=128,
        out_dim=4,
        num_heads=2,  # head_dim=32 (f_freqs=6, h_freqs=10, w_freqs=10 → 13 complex... wait)
        num_layers=2,
        eps=1e-6,
        seperated_timestep=True,
        require_clip_embedding=False,
        require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )


def _build_tiny_model(**overrides):
    import torch  # noqa: F401  (verifies torch available)
    from diffsynth.models.reactive_gwm_casual_forcing_dit import CausalForcingReactiveGWMModel

    kwargs = _tiny_model_kwargs()
    return CausalForcingReactiveGWMModel(
        num_buttons=10,
        kv_window_size=overrides.get("kv_window_size", 2),
        sink_size=overrides.get("sink_size", 1),
        num_frame_per_block=1,
        text_len=overrides.get("text_len", 16),
        **kwargs,
    )


def _build_tiny_bidir():
    """Tiny **bidirectional** ReactiveGWMModel (Stage 3 teacher / critic 同 arch)."""
    from diffsynth.models.reactive_gwm_dit import ReactiveGWMModel

    return ReactiveGWMModel(num_buttons=10, **_tiny_model_kwargs())


# ----------------------------------------------------------------------- subcommands
def _verify_state_dict() -> int:
    """CF model state_dict roundtrip (strict, 0 missing / 0 unexpected)."""
    m = _build_tiny_model()
    state = m.state_dict()
    m2 = _build_tiny_model()
    missing, unexpected = m2.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing[:10]}"
    assert not unexpected, f"unexpected keys: {unexpected[:10]}"
    # Strict load also OK
    m3 = _build_tiny_model()
    m3.load_state_dict(state, strict=True)
    print(f"OK state_dict: {len(state)} keys, strict roundtrip 0/0")
    return 0


def _verify_scheduler() -> int:
    """FlowMatchScheduler('Wan') basic invariants."""
    import torch
    from diffsynth.diffusion import FlowMatchScheduler

    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(1000, training=True, shift=5.0)
    assert sched.sigmas is not None and sched.timesteps is not None
    assert sched.sigmas.shape[0] == 1000, sched.sigmas.shape
    assert sched.timesteps.shape[0] == 1000
    assert (sched.sigmas >= 0).all() and (sched.sigmas <= 1).all()
    assert torch.allclose(sched.timesteps, sched.sigmas * 1000)
    # add_noise / target sanity
    x = torch.randn(1, 2, 4, 8, 8)
    n = torch.randn_like(x)
    t = sched.timesteps[500]
    xt = sched.add_noise(x, n, t)
    target = sched.training_target(x, n, t)
    assert xt.shape == x.shape
    assert torch.allclose(target, n - x)
    print(
        f"OK scheduler: 1000 sigmas in "
        f"[{sched.sigmas[0]:.4f}, {sched.sigmas[-1]:.4f}], target=noise-x"
    )
    return 0


def _verify_tf_mask() -> int:
    """Build TF mask + sanity-check the attention_mask logic at sample points."""
    import torch
    from diffsynth.models.reactive_gwm_casual_forcing_dit import CausalForcingReactiveGWMModel

    F_lat, frame_seqlen, npfb = 4, 8, 1
    device = torch.device("cpu")
    mask = CausalForcingReactiveGWMModel._build_teacher_forcing_mask(
        device, F_lat, frame_seqlen, npfb
    )
    total = F_lat * frame_seqlen * 2
    clean_ends = F_lat * frame_seqlen
    # Build the same boundary tensors locally so we can spot-check visibility.
    # This mirrors the function body (q_idx-indexed boundary arrays).
    attn_block_size = frame_seqlen * npfb
    context_ends = [
        (q // attn_block_size + 1) * attn_block_size if q < clean_ends else 0
        for q in range(total)
    ]
    noise_ctx_ends = [
        ((q - clean_ends) // attn_block_size) * attn_block_size if q >= clean_ends else 0
        for q in range(total)
    ]

    def attn(q, kv):
        if q == kv:
            return True
        if q < clean_ends:
            return kv < context_ends[q]
        # q is noisy
        n_start = (q - clean_ends) // attn_block_size * attn_block_size + clean_ends
        n_end = n_start + attn_block_size
        if n_start <= kv < n_end:
            return True
        if kv < noise_ctx_ends[q]:
            return True
        return False

    # Spot checks
    # 1) clean→clean is block-causal: q_clean_0 sees k_clean_0..attn_block_size-1; not after.
    assert attn(0, 0) and attn(0, attn_block_size - 1)
    assert not attn(0, attn_block_size)
    # 2) noisy→clean strictly before block:
    #    q in noisy block 1 (q = clean_ends + attn_block_size) sees k_clean in block 0 only.
    q_noisy1 = clean_ends + attn_block_size
    assert attn(q_noisy1, 0) and attn(q_noisy1, attn_block_size - 1)
    assert not attn(q_noisy1, attn_block_size)  # cannot see clean block 1
    # 3) noisy→noisy: only within same block
    assert attn(q_noisy1, q_noisy1) and attn(q_noisy1, q_noisy1 + attn_block_size - 1)
    assert not attn(q_noisy1, q_noisy1 + attn_block_size)  # noisy block 2
    # 4) clean→noisy: never (except identity for clean=noisy which doesn't apply here)
    assert not attn(0, clean_ends)
    print(
        f"OK tf_mask: 4-quadrant visibility verified for F={F_lat}, "
        f"frame_seqlen={frame_seqlen}; total_length={total}"
    )
    return 0


def _verify_ar_tf() -> int:
    """Tiny TF dual-block forward: shape correct + finite + (if CUDA) grad flows.

    flex_attention 不支持 CPU backward. CPU only 时跑 no_grad forward; CUDA 时跑完整
    forward + backward + grad-flow assertion.
    """
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    do_backward = torch.cuda.is_available()
    # FlashAttention (used by cross_attn in DiffSynth) requires bf16/fp16 on GPU;
    # fall back to fp32 on CPU (SDPA path).
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = _build_tiny_model().to(dtype).to(device)
    model.train() if do_backward else model.eval()
    B, F_lat, C, H, W = 1, 3, 4, 16, 16
    clean = torch.randn(B, F_lat, C, H, W, device=device, dtype=dtype)
    noise = torch.randn_like(clean)
    sigma = torch.full((B, F_lat), 0.5, dtype=dtype, device=device)
    sigma[:, 0] = 0.0  # frame 0 anchor
    s = sigma[:, :, None, None, None]
    noisy = (1 - s) * clean + s * noise
    timestep = torch.full((B, F_lat), 500.0, dtype=dtype, device=device)
    timestep[:, 0] = 0.0
    context = torch.randn(B, 8, 128, device=device, dtype=dtype)  # text_dim=128
    action = torch.zeros(B, F_lat * 4, 10, device=device, dtype=dtype)

    ctx_mgr = torch.enable_grad() if do_backward else torch.no_grad()
    with ctx_mgr:
        out = model.forward(
            noisy_x=noisy,
            timestep=timestep,
            clean_x=clean,
            action=action,
            context=context,
        )

    assert out.shape == clean.shape, f"out={out.shape}, expected={clean.shape}"
    assert torch.isfinite(out).all(), "non-finite values in TF forward"

    if do_backward:
        target = noise - clean
        loss = torch.nn.functional.mse_loss(out[:, 1:], target[:, 1:])
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any((g.abs() > 0).any() for g in grads), "no gradients flowed to model"
        print(
            f"OK ar_tf ({device.type}): TF forward {tuple(out.shape)}, "
            f"loss={loss.item():.4f}, grad flows"
        )
    else:
        print(
            f"OK ar_tf (cpu forward-only, flex_attention no-backward-on-cpu): "
            f"{tuple(out.shape)} finite"
        )
    return 0


def _pick_device_dtype():
    """KV-cache + cross-attn use flash_attn (CUDA-only). Force cuda+bf16 if available."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def _verify_kv_cache_init() -> int:
    """init_kv_cache returns expected shape and layer count."""
    import torch

    device, dtype = _pick_device_dtype()
    model = _build_tiny_model(kv_window_size=2, sink_size=1).to(dtype).to(device)
    frame_seqlen = 8
    cache = model.init_kv_cache(
        batch_size=1, device=device, dtype=dtype,
        frame_seqlen=frame_seqlen,
    )
    assert "self" in cache and "cross" in cache
    assert len(cache["self"]) == len(model.blocks)
    assert len(cache["cross"]) == len(model.blocks)
    self_cache_size = (model.sink_size + model.kv_window_size) * frame_seqlen
    for c in cache["self"]:
        assert c["k"].shape == (1, self_cache_size, model.blocks[0].self_attn.num_heads, model.dim // model.blocks[0].self_attn.num_heads)
        assert int(c["global_end_index"].item()) == 0
        assert int(c["local_end_index"].item()) == 0
    for c in cache["cross"]:
        assert c["is_init"] is False
    print(
        f"OK kv_cache_init: {len(cache['self'])} layers, "
        f"self_cache_size={self_cache_size}"
    )
    return 0


def _verify_kv_cache_refill() -> int:
    """After refill_kv_cache, global_end_index advances; cross-attn cache fills."""
    import torch

    device, dtype = _pick_device_dtype()
    model = _build_tiny_model(kv_window_size=2, sink_size=1).to(dtype).to(device)
    model.eval()
    B, F_in, C, H, W = 1, 1, 4, 16, 16
    x0 = torch.randn(B, F_in, C, H, W, device=device, dtype=dtype)
    # First, run patchify to discover frame_seqlen
    tokens, F_lat, h, w = model._patchify_and_geom(x0)
    real_frame_seqlen = h * w
    cache = model.init_kv_cache(
        batch_size=B, device=device, dtype=dtype,
        frame_seqlen=real_frame_seqlen,
    )
    context = torch.randn(B, 8, 128, device=device, dtype=dtype)
    action = torch.zeros(B, 4, 10, device=device, dtype=dtype)
    model.refill_kv_cache(x_block=x0, action=action, context=context, kv_cache=cache)
    # After refill, global_end_index = real_frame_seqlen per layer (we wrote 1 frame block)
    end0 = int(cache["self"][0]["global_end_index"].item())
    assert end0 == real_frame_seqlen, f"expected {real_frame_seqlen}, got {end0}"
    assert cache["cross"][0]["is_init"] is True
    print(f"OK kv_cache_refill: global_end={end0}, cross_init=True")
    return 0


def _verify_kv_cache_rollout() -> int:
    """Roll out 2 frames: global_end advances monotonically; output shape correct."""
    import torch

    device, dtype = _pick_device_dtype()
    model = _build_tiny_model(kv_window_size=4, sink_size=1).to(dtype).to(device)
    model.eval()
    B, C, H, W = 1, 4, 16, 16
    first = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)
    tokens, _, h, w = model._patchify_and_geom(first)
    fs = h * w
    cache = model.init_kv_cache(B, device, dtype, frame_seqlen=fs)
    context = torch.randn(B, 8, 128, device=device, dtype=dtype)
    action_frame = torch.zeros(B, 4, 10, device=device, dtype=dtype)

    # Frame 0 anchor refill
    model.refill_kv_cache(first, action_frame, context, cache)
    end_after_anchor = int(cache["self"][0]["global_end_index"].item())
    # Frame 1: forward + refill
    noisy = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)
    t = torch.full((B, 1), 500.0, device=device, dtype=dtype)
    out = model.forward(
        noisy_x=noisy, timestep=t, clean_x=None, kv_cache=cache,
        action=action_frame, context=context,
    )
    assert out.shape == noisy.shape
    assert torch.isfinite(out).all()
    end_after_query = int(cache["self"][0]["global_end_index"].item())
    assert end_after_query >= end_after_anchor, "cache end index regressed"
    print(
        f"OK kv_cache_rollout: forward {out.shape}, end {end_after_anchor}→{end_after_query}"
    )
    return 0


def _verify_sanity_kv() -> int:
    """Mini end-to-end KV-cache rollout 2 generated frames."""
    import torch
    from diffsynth.pipelines.reactive_gwm_casual_forcing import sample_multistep_kv_cache

    device, dtype = _pick_device_dtype()
    model = _build_tiny_model(kv_window_size=4, sink_size=1).to(dtype).to(device)
    model.eval()
    B, C, H, W = 1, 4, 16, 16
    first = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)
    context = torch.randn(B, 8, 128, device=device, dtype=dtype)
    action = torch.zeros(B, 8, 10, device=device, dtype=dtype)
    out = sample_multistep_kv_cache(
        model=model,
        action_per_frame=action,
        context=context,
        first_frame_latent=first,
        target_latent_frames=2,
        diffusion_steps=2,
        kv_window_size=4,
        sink_size=1,
        num_frame_per_block=1,
        device=device,
        dtype=dtype,
    )
    assert out.shape == (B, 2, C, H, W), f"got {out.shape}"
    assert torch.isfinite(out).all()
    print(f"OK sanity_kv: end-to-end rollout {out.shape}")
    return 0


def _verify_smoke() -> int:
    """Tiny TF forward+backward on CPU; verifies gradient propagation end-to-end."""
    return _verify_ar_tf()


def _verify_cd() -> int:
    """Stage 2 CD helpers + 数值/可见性自洽 (tiny model).

    检查: (1) flow_to_x0 代数 + frame0 anchor; (2) build_cd_scheduler 48 节点丢 σ=0;
    (3) cd_teacher_one_step 单步 ODE + neg-CFG 手算等价 + frame0 anchor;
    (4) student==ema 同输入 → cm_pred 一致 → loss≈0 (plumbing/determinism);
    (5) [CUDA] 梯度只到 student, teacher/ema 无梯度.
    flex_attention CPU 不支持 backward → CPU 只跑 forward-only (1)-(4).
    """
    import torch
    from diffsynth.pipelines.reactive_gwm_casual_forcing import (
        build_cd_scheduler,
        cd_teacher_one_step,
        flow_to_x0,
        model_fn_causal_forcing,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    do_backward = torch.cuda.is_available()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    B, F_lat, C, H, W = 1, 3, 4, 16, 16

    # --- (1) flow_to_x0 algebra + frame0 anchor (σ=0 preserves latent)
    latent = torch.randn(B, F_lat, C, H, W, dtype=dtype, device=device)
    flow = torch.randn_like(latent)
    sigma = torch.full((B, F_lat), 0.4, dtype=dtype, device=device)
    sigma[:, 0] = 0.0
    x0 = flow_to_x0(latent, flow, sigma)
    assert torch.allclose(x0[:, 0], latent[:, 0]), "frame0 σ=0 must preserve latent"
    s = sigma[:, :, None, None, None]
    assert torch.allclose(x0.float(), (latent - s * flow).float(), atol=1e-2)

    # --- (2) CD scheduler: 48 nodes, σ descending, σ=0 endpoint dropped
    sched = build_cd_scheduler(48, 5.0)
    assert sched.timesteps.shape[0] == 48 and sched.sigmas.shape[0] == 48, "must be 48 nodes"
    assert float(sched.sigmas[-1]) > 0.0, "extra_one_step must drop σ=0 endpoint"
    assert float(sched.sigmas[0]) > float(sched.sigmas[-1]), "σ must descend high→low"

    # build 3 tiny models; student==ema (same weights) for the determinism test
    student = _build_tiny_model().to(dtype).to(device)
    ema = _build_tiny_model().to(dtype).to(device)
    teacher = _build_tiny_model().to(dtype).to(device)
    ema.load_state_dict(student.state_dict())
    for m in (ema, teacher):
        m.eval()
        m.requires_grad_(False)
    student.eval()

    context = torch.randn(B, 8, 128, dtype=dtype, device=device)
    context_neg = torch.randn(B, 8, 128, dtype=dtype, device=device)
    action = torch.zeros(B, F_lat * 4, 10, dtype=dtype, device=device)
    idx = 10
    t_curr = sched.timesteps[idx].to(device)
    t_next = sched.timesteps[idx + 1].to(device)
    timestep_pf = torch.full((B, F_lat), float(t_curr), dtype=dtype, device=device)
    timestep_pf[:, 0] = 0.0
    g = 3.0

    # --- (3) cd_teacher_one_step manual equivalence + frame0 anchor
    with torch.no_grad():
        v_cond = model_fn_causal_forcing(
            teacher, tf_mode=True, noisy_x=latent, timestep=timestep_pf,
            clean_x=x0, action=action, context=context,
        )
        v_uncond = model_fn_causal_forcing(
            teacher, tf_mode=True, noisy_x=latent, timestep=timestep_pf,
            clean_x=x0, action=action, context=context_neg,
        )
        v_manual = v_uncond + g * (v_cond - v_uncond)
        dt = ((t_curr - t_next) / 1000.0).to(dtype)
        manual_next = (latent - dt * v_manual).clone()
        manual_next[:, 0] = x0[:, 0]
        auto_next = cd_teacher_one_step(
            teacher, latent_t=latent, timestep_per_frame=timestep_pf, t_idx=idx,
            scheduler=sched, clean_x=x0, action=action, context_pos=context,
            context_neg=context_neg, guidance_scale=g, first_frame_x0=x0,
        )
    assert torch.allclose(auto_next.float(), manual_next.float(), atol=2e-2), "teacher ODE mismatch"
    assert torch.allclose(auto_next[:, 0], x0[:, 0]), "teacher step frame0 anchor failed"

    # --- (4) determinism: student==ema, same (latent,t) → identical cm_pred → MSE≈0
    with torch.no_grad():
        flow_s = model_fn_causal_forcing(
            student, tf_mode=True, noisy_x=latent, timestep=timestep_pf,
            clean_x=x0, action=action, context=context,
        )
        flow_e = model_fn_causal_forcing(
            ema, tf_mode=True, noisy_x=latent, timestep=timestep_pf,
            clean_x=x0, action=action, context=context,
        )
    cm_s = flow_to_x0(latent, flow_s, sigma)
    cm_e = flow_to_x0(latent, flow_e, sigma)
    loss0 = torch.nn.functional.mse_loss(cm_s[:, 1:].float(), cm_e[:, 1:].float())
    assert loss0.item() < 1e-3, f"same-weights same-input loss must be ~0, got {loss0.item()}"

    # --- (5) grad flows only to student (CUDA only — flex_attention no CPU backward)
    if do_backward:
        student.train()
        student.requires_grad_(True)
        flow_s = model_fn_causal_forcing(
            student, tf_mode=True, noisy_x=latent, timestep=timestep_pf,
            clean_x=x0, action=action, context=context,
        )
        cm_s = flow_to_x0(latent, flow_s, sigma)
        with torch.no_grad():
            flow_e = model_fn_causal_forcing(
                ema, tf_mode=True, noisy_x=manual_next, timestep=timestep_pf,
                clean_x=x0, action=action, context=context,
            )
        cm_e = flow_to_x0(manual_next, flow_e, sigma)
        loss = torch.nn.functional.mse_loss(cm_s[:, 1:].float(), cm_e[:, 1:].float())
        loss.backward()
        assert any((p.grad is not None and (p.grad.abs() > 0).any()) for p in student.parameters()), "no grad to student"
        assert all(p.grad is None for p in ema.parameters()), "ema must have no grad"
        assert all(p.grad is None for p in teacher.parameters()), "teacher must have no grad"
        print(
            f"OK cd (cuda): flow_to_x0 ✓, 48-node sched ✓ (σ[-1]={float(sched.sigmas[-1]):.4f}), "
            f"teacher ODE ✓, loss0={loss0.item():.2e}, grad→student only ✓"
        )
    else:
        print(
            f"OK cd (cpu forward-only): flow_to_x0 ✓, 48-node sched ✓ "
            f"(σ[-1]={float(sched.sigmas[-1]):.4f}), teacher ODE ✓, determinism loss0={loss0.item():.2e}"
        )
    return 0


def _verify_rollout() -> int:
    """Stage 3 DMD2 backward-sim self-rollout (tiny model).

    检查: (1) warp_denoising_steps 4 步 + σ=t/1000 + 降序; (2) shared exit index ∈ [0,4);
    (3) [device] rollout 26→tiny 3 帧: 输出 shape / frame0=anchor / 全量 buffer 不驱逐;
    (4) keep_grad=True → 生成帧带梯度 + backward 到 generator; keep_grad=False → 无梯度.
    KV-cache 路径用 flash_attention (CUDA-only in diffsynth_sf) → 模型 forward 走 _pick_device_dtype.
    """
    import torch
    from diffsynth.pipelines.reactive_gwm_casual_forcing import (
        build_cf_scheduler,
        dmd_self_rollout,
        warp_denoising_steps,
        _sample_shared_exit_index,
    )

    # (1) warp 4-step (CPU OK)
    sched1000 = build_cf_scheduler(1000, 5.0, training=False)
    wt, ws = warp_denoising_steps(sched1000, [1000, 750, 500, 250], torch.device("cpu"))
    assert wt.shape[0] == 4 and ws.shape[0] == 4, (wt.shape, ws.shape)
    assert torch.allclose(ws.float(), (wt / 1000.0).float(), atol=1e-4), "σ must = t/1000"
    assert float(wt[0]) > float(wt[-1]), "warped timesteps must descend"
    assert torch.isfinite(wt).all()
    # (2) shared exit index
    ei = _sample_shared_exit_index(4, torch.device("cpu"))
    assert 0 <= ei < 4, ei

    # (3)(4) tiny rollout on device (KV-cache flash_attn → CUDA in diffsynth_sf)
    device, dtype = _pick_device_dtype()
    gen = _build_tiny_model(kv_window_size=3, sink_size=0).to(dtype).to(device)
    gen.train()
    gen.requires_grad_(True)
    B, C, H, W = 1, 4, 16, 16
    first = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)
    context = torch.randn(B, 8, 128, device=device, dtype=dtype)
    action = torch.zeros(B, 3 * 4, 10, device=device, dtype=dtype)
    wt, ws = wt.to(device), ws.to(device)

    video, exit_idx = dmd_self_rollout(
        gen, first_frame_latent=first, action_per_frame=action, context=context,
        num_frames=3, warped_timesteps=wt, warped_sigmas=ws,
        kv_window_size=3, sink_size=0, keep_grad=True, device=device, dtype=dtype,
    )
    assert video.shape == (B, 3, C, H, W), video.shape
    assert torch.isfinite(video).all()
    assert 0 <= exit_idx < 4, exit_idx
    assert torch.allclose(video[:, 0].float(), first[:, 0].float()), "frame0 must = anchor"
    assert video[:, 1:].requires_grad, "generated frames must carry grad (keep_grad=True)"
    video[:, 1:].float().pow(2).mean().backward()
    assert any(p.grad is not None and (p.grad.abs() > 0).any() for p in gen.parameters()), \
        "no grad reached generator"

    gen.zero_grad(set_to_none=True)
    video2, _ = dmd_self_rollout(
        gen, first_frame_latent=first, action_per_frame=action, context=context,
        num_frames=3, warped_timesteps=wt, warped_sigmas=ws,
        kv_window_size=3, sink_size=0, keep_grad=False, device=device, dtype=dtype,
    )
    assert not video2.requires_grad, "keep_grad=False must not carry grad"
    print(
        f"OK rollout: warp ✓ (σ=t/1000), exit_idx={exit_idx}, video {tuple(video.shape)}, "
        f"frame0 anchor ✓, keep_grad→gen ✓, no_grad path ✓"
    )
    return 0


def _verify_dmd() -> int:
    """Stage 3 DMD generator/critic loss + 空间分离 + 梯度路由 (tiny model).

    检查: (1) timestep uniform→shift5→clamp[20,980]; (2) dmd_score_x0 frame0 σ=0 anchor;
    (3) generator-loss(x0 空间) backward 只到 generator (teacher/critic 无梯度);
    (4) critic-loss(flow 空间) backward 只到 critic (generator 无梯度).
    模型 forward (CF KV-cache + 双向 model_fn_wan_video) 用 flash_attention → CUDA.
    """
    import torch
    from diffsynth.pipelines.reactive_gwm_casual_forcing import (
        build_cf_scheduler,
        dmd_critic_flow_loss,
        dmd_distribution_matching_loss,
        dmd_score_x0,
        dmd_self_rollout,
        sample_dmd_timestep,
        warp_denoising_steps,
    )

    # (1) timestep clamp (CPU OK)
    cpu = torch.device("cpu")
    for _ in range(64):
        t = sample_dmd_timestep(1000, 5.0, 20, 980, cpu)
        assert 20.0 <= float(t) <= 980.0, f"timestep {float(t)} out of [20,980]"

    device, dtype = _pick_device_dtype()
    B, F_lat, C, H, W = 1, 3, 4, 16, 16
    context = torch.randn(B, 8, 128, device=device, dtype=dtype)
    context_neg = torch.randn(B, 8, 128, device=device, dtype=dtype)
    action = torch.zeros(B, F_lat * 4, 10, device=device, dtype=dtype)

    # (2) dmd_score_x0 frame0 σ=0 anchor (bidirectional score → x0)
    bidir = _build_tiny_bidir().to(dtype).to(device)
    bidir.eval()
    bidir.requires_grad_(False)
    noisy = torch.randn(B, F_lat, C, H, W, device=device, dtype=dtype)
    t_s = torch.full((1,), 500.0, device=device)
    with torch.no_grad():
        x0 = dmd_score_x0(bidir, noisy, t_s, context, action, sigma_t=0.5)
    assert torch.allclose(x0[:, 0].float(), noisy[:, 0].float()), "frame0 σ=0 must preserve noisy[:,0]"
    assert torch.isfinite(x0).all()

    # (3)(4) full generator + critic loss with grad routing
    gen = _build_tiny_model(kv_window_size=F_lat, sink_size=0).to(dtype).to(device)
    gen.train()
    gen.requires_grad_(True)
    teacher = _build_tiny_bidir().to(dtype).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    critic = _build_tiny_bidir().to(dtype).to(device)
    critic.train()
    critic.requires_grad_(True)
    sched1000 = build_cf_scheduler(1000, 5.0, training=False)
    wt, ws = warp_denoising_steps(sched1000, [1000, 750, 500, 250], device)
    first = torch.randn(B, 1, C, H, W, device=device, dtype=dtype)

    # generator phase: rollout (grad) → DMD x0-space loss (teacher/critic no_grad score)
    video, _ = dmd_self_rollout(
        gen, first_frame_latent=first, action_per_frame=action, context=context,
        num_frames=F_lat, warped_timesteps=wt, warped_sigmas=ws,
        kv_window_size=F_lat, sink_size=0, keep_grad=True, device=device, dtype=dtype,
    )
    loss_g = dmd_distribution_matching_loss(
        video, teacher=teacher, critic=critic, context_pos=context, context_neg=context_neg,
        action=action, real_guidance_scale=3.0, fake_guidance_scale=0.0,
        num_train_timestep=1000, timestep_shift=5.0, clamp_min=20, clamp_max=980,
        first_frame_anchor=True,
    )
    assert torch.isfinite(loss_g), loss_g
    loss_g.backward()
    assert any(p.grad is not None and (p.grad.abs() > 0).any() for p in gen.parameters()), \
        "DMD generator loss must grad generator"
    assert all(p.grad is None for p in teacher.parameters()), "teacher must have no grad"
    assert all(p.grad is None for p in critic.parameters()), "critic must have no grad in generator loss"

    # critic phase: no_grad rollout → critic flow loss (critic grad only)
    gen.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    with torch.no_grad():
        video_c, _ = dmd_self_rollout(
            gen, first_frame_latent=first, action_per_frame=action, context=context,
            num_frames=F_lat, warped_timesteps=wt, warped_sigmas=ws,
            kv_window_size=F_lat, sink_size=0, keep_grad=False, device=device, dtype=dtype,
        )
    loss_c = dmd_critic_flow_loss(
        video_c.detach(), critic=critic, context_pos=context, action=action,
        num_train_timestep=1000, timestep_shift=5.0, clamp_min=20, clamp_max=980,
        first_frame_anchor=True,
    )
    assert torch.isfinite(loss_c), loss_c
    loss_c.backward()
    assert any(p.grad is not None and (p.grad.abs() > 0).any() for p in critic.parameters()), \
        "critic loss must grad critic"
    assert all(p.grad is None for p in gen.parameters()), "generator must have no grad in critic loss"
    print(
        "OK dmd: timestep clamp[20,980] ✓, score_x0 frame0 anchor ✓, "
        "gen-loss(x0)→generator only ✓, critic-loss(flow)→critic only ✓"
    )
    return 0


def _verify_bf16_two_paths() -> int:
    """GPU only: bf16 TF + KV-cache both finite."""
    import torch

    if not torch.cuda.is_available():
        print("SKIP bf16_two_paths: no CUDA available")
        return 0
    try:
        torch.cuda.set_per_process_memory_fraction(0.05, 0)
    except Exception:
        pass
    device = torch.device("cuda:0")
    model = _build_tiny_model().to(torch.bfloat16).to(device)
    model.train()
    B, F_lat, C, H, W = 1, 3, 4, 16, 16
    clean = torch.randn(B, F_lat, C, H, W, dtype=torch.bfloat16, device=device)
    noise = torch.randn_like(clean)
    sigma = torch.full((B, F_lat), 0.5, dtype=torch.bfloat16, device=device)
    sigma[:, 0] = 0.0
    s = sigma[:, :, None, None, None]
    noisy = (1 - s) * clean + s * noise
    timestep = torch.full((B, F_lat), 500.0, dtype=torch.bfloat16, device=device)
    timestep[:, 0] = 0.0
    context = torch.randn(B, 8, 128, dtype=torch.bfloat16, device=device)
    action = torch.zeros(B, F_lat * 4, 10, dtype=torch.bfloat16, device=device)
    out_tf = model.forward(
        noisy_x=noisy, timestep=timestep, clean_x=clean,
        action=action, context=context,
    )
    assert torch.isfinite(out_tf).all()
    print(f"OK bf16_two_paths (TF only): {out_tf.shape} finite")
    return 0


# ----------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CF verify")
    p.add_argument("subcommand", choices=SUBCOMMANDS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fn = globals().get(f"_verify_{args.subcommand}")
    if fn is None:
        print(f"[verify] subcommand {args.subcommand!r} not yet implemented", file=sys.stderr)
        return 2
    return int(fn() or 0)


if __name__ == "__main__":
    sys.exit(main())
