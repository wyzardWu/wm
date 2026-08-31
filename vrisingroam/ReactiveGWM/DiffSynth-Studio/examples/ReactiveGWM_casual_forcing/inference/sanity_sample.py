"""Sanity-check KV-cache 自回归推理 — Stage 1+ 训练中 spawn 的 subprocess 入口.

算法 (对齐 CF 上游 `pipeline/causal_diffusion_inference.py`):
  1. 加载 CausalForcingReactiveGWMModel + VAE + T5 (复用 ReactiveGWMPipeline.from_pretrained)
  2. T5 编码 fixed prompt → cross-attn cache (一次性, by refill_kv_cache 触发)
  3. VAE 编码首帧 → first_frame_latent (anchor)
  4. refill_kv_cache(first_frame_latent, t=0) → 首帧 K/V 入 cache (cache-only)
  5. 逐 latent 帧 rollout i = 1 .. target_latent_frames:
       a) noisy_x = randn [1, 1, C, H, W]
       b) multi-step flow-match (默认 50 步) 去噪
       c) refill_kv_cache(x0_i, t=0) → 新帧 K/V 入 cache (sink+recent eviction)
  6. VAE 解码所有 latent → ffmpeg mp4 写盘 (25 FPS, ≈60s)

Stage 1/2 用 50 步; Stage 3 yaml 覆盖 `sanity_sample_diffusion_steps: 4`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# sys.path setup so subprocess (which inherits parent env vars) can import.
_HERE = Path(__file__).resolve().parent.parent  # examples/ReactiveGWM_casual_forcing/
_REPO_ROOT = _HERE.parent.parent  # DiffSynth-Studio/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_REACTIVE_GWM = _HERE.parent / "ReactiveGWM"
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CF sanity-check KV-cache rollout")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="gpu_shared", choices=["gpu_shared", "cpu"])
    p.add_argument("--seed", type=int, default=None,
                   help="固定 rollout 噪声种子 (默认 None=不固定, 保持 in-training spawn 行为不变); 离线公平对比用")
    p.add_argument("--fill_action", default="idle", choices=["idle", "light_punch", "heavy_kick"],
                   help="how to pad action beyond the clip's real length")
    p.add_argument("--action_sequence", default=None,
                   help="comma-separated action aliases to repeat over the whole rollout, e.g. jump,light_punch,heavy_kick")
    p.add_argument("--action_segment_frames", type=int, default=None,
                   help="pixel frames to hold each action_sequence item; default=profile action_hold_window")
    p.add_argument("--output_fps", type=int, default=20,
                   help="FPS used when writing the output mp4")
    p.add_argument("--memory_fraction", type=float, default=None,
                   help="覆盖 cfg.sanity_sample_memory_fraction (按总显存比例的进程上限); "
                        "默认 None=沿用 cfg (in-training spawn 行为不变). 离线共享卡建议调高如 0.35")
    p.add_argument("--diffusion_steps", type=int, default=None,
                   help="覆盖 cfg.sanity_sample_diffusion_steps (每帧 multi-step 去噪步数); "
                        "默认 None=沿用 cfg (Stage1/2=50). 离线提速可降到 30/10 (质量略糊)")
    p.add_argument("--sampler", default=None, choices=["ode", "dmd"],
                   help="覆盖 cfg.sanity_sample_sampler: ode=flow ODE sampler, dmd=DMD re-noise sampler")
    p.add_argument("--first_frame_image", default=None,
                   help="custom seed image (png); together with --actions_parquet bypasses load_fixed_sanity")
    p.add_argument("--actions_parquet", default=None,
                   help="per-frame action parquet (profile button cols selected)")
    p.add_argument("--latent_frames", type=int, default=None,
                   help="override cfg.sanity_sample_latent_frames")
    return p.parse_args()


def _load_cfg(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    default_path = Path(yaml_path).parent / "default.yaml"
    if default_path.exists() and default_path.resolve() != Path(yaml_path).resolve():
        with open(default_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def _setup_device(device_mode: str, cfg: dict[str, Any], frac_override: float | None = None) -> str:
    if device_mode == "cpu":
        return "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Reduce fragmentation under the small fraction cap.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        frac = float(frac_override) if frac_override is not None \
            else float(cfg.get("sanity_sample_memory_fraction", 0.2))
        try:
            torch.cuda.set_per_process_memory_fraction(frac, 0)
        except Exception as e:  # noqa: BLE001
            print(f"[Sanity] set_per_process_memory_fraction failed: {e}", flush=True)
        return "cuda:0"
    return "cpu"


def _to_gpu(model: torch.nn.Module, device: str) -> torch.nn.Module:
    model.to(device)
    return model


def _to_cpu(model: torch.nn.Module) -> torch.nn.Module:
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def main() -> int:
    args = parse_args()
    cfg = _load_cfg(args.config)
    device = _setup_device(args.device, cfg, args.memory_fraction)
    print(f"[Sanity] device={device} ckpt={args.ckpt}"
          + (f" mem_frac={args.memory_fraction}" if args.memory_fraction is not None else ""),
          flush=True)

    # Imports kept lazy so --help works without torch/CUDA dependencies installed.
    from safetensors.torch import load_file
    from examples.ReactiveGWM_casual_forcing.inference.fixed_sanity import load_fixed_sanity
    from examples.ReactiveGWM_casual_forcing.modules.ar_tf import (
        WAN_MODEL_KWARGS,
        _build_model_configs,
    )
    from data.profiles import get_profile
    from diffsynth.core import ModelConfig
    from diffsynth.models.reactive_gwm_casual_forcing_dit import CausalForcingReactiveGWMModel
    from diffsynth.pipelines.reactive_gwm import ReactiveGWMPipeline
    from diffsynth.pipelines.reactive_gwm_casual_forcing import (
        sample_dmd_kv_cache,
        sample_multistep_kv_cache,
    )

    profile = get_profile(cfg.get("game", "sf3"))

    # 1) Fixed sanity sample (prompt + action + first_frame video path)
    if args.first_frame_image and args.actions_parquet:
        import pandas as pd
        _adf = pd.read_parquet(args.actions_parquet)
        _cols = list(profile.button_cols)
        _act = torch.tensor(_adf[_cols].values, dtype=torch.float32).unsqueeze(0)
        sd = {"prompt": profile.fixed_prompt, "action": _act,
              "video_path": args.first_frame_image}
        print(f"[Sanity] custom eval inputs: image={args.first_frame_image} "
              f"actions={args.actions_parquet} T={_act.shape[1]}", flush=True)
    else:
        sd = load_fixed_sanity(
            metadata_path=cfg["metadata_path"],
            dataset_base=cfg["dataset_base"],
            game=cfg.get("game", "sf3"),
            clip_idx=int(cfg.get("sanity_sample_clip_idx", 0)),
            target_pixel_frames=int(cfg.get("sanity_sample_pixel_frames", 1500)),
            use_csv_prompt=bool(cfg.get("use_csv_prompt", True)),
            prompt_column=cfg.get("prompt_column", "prompt"),
            fill_action=args.fill_action,
            action_sequence=args.action_sequence,
            action_segment_frames=args.action_segment_frames,
        )
    action_desc = f"action_sequence={args.action_sequence}" if args.action_sequence else f"fill_action={args.fill_action}"
    print(f"[Sanity] prompt={sd['prompt'][:80]!r} {action_desc}", flush=True)

    # 2) Build pipe entirely on CPU (T5 + VAE + base Wan DiT all on RAM);
    #    we manually swap one model at a time to GPU below to cap peak VRAM.
    model_configs = _build_model_configs(cfg)
    pipe = ReactiveGWMPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=cfg["tokenizer_dir"]),
    )
    # Restore pipe.device so helpers (preprocess_image, etc.) put tensors on GPU
    # while module weights stay on CPU until each phase moves them up.
    pipe.device = device

    # Replace base WanModel with CausalForcing variant (CPU first; ckpt also on CPU).
    custom_dit = CausalForcingReactiveGWMModel(
        num_buttons=profile.num_buttons,
        kv_window_size=int(cfg.get("sanity_sample_kv_window_size", 16)),
        sink_size=int(cfg.get("sanity_sample_sink_size", 2)),
        num_frame_per_block=int(cfg.get("num_frame_per_block", 1)),
        text_len=int(cfg.get("text_len", 512)),
        **WAN_MODEL_KWARGS,
    ).to(torch.bfloat16)
    state = load_file(args.ckpt)
    missing, unexpected = custom_dit.load_state_dict(state, strict=False)
    print(f"[Sanity] ckpt load: {len(state)} keys, missing={len(missing)}, unexpected={len(unexpected)}",
          flush=True)
    del state
    del pipe.dit
    pipe.dit = custom_dit  # CPU
    pipe.dit.eval()
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.empty_cache()

    # 3) Encode prompt(s) — swap T5 to GPU, run, swap back to CPU.
    #    CFG (对齐 CF 上游 causal_diffusion_inference.py): 同时编码 negative prompt,
    #    rollout 每步 pos/neg 两路 forward + guidance_scale 合成 (见 sample_multistep_kv_cache).
    from diffsynth.pipelines.reactive_gwm_casual_forcing import DEFAULT_NEGATIVE_PROMPT

    _to_gpu(pipe.text_encoder, device)

    def _encode_prompt(prompt: str) -> torch.Tensor:
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(device), mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        ctx = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            ctx[:, v:] = 0
        return ctx.to(torch.bfloat16).to(device)

    context = _encode_prompt(sd["prompt"])
    guidance_scale = float(cfg.get("sanity_sample_guidance_scale", 3.0))
    neg_prompt = cfg.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
    context_neg = _encode_prompt(neg_prompt) if guidance_scale != 1.0 else None
    _to_cpu(pipe.text_encoder)

    # 4) VAE-encode first frame from sanity clip video — swap VAE to GPU.
    from PIL import Image
    import imageio.v3 as iio

    from diffsynth.core.data.operators import ImageCropAndResize

    if args.first_frame_image:
        first_frame_arr = iio.imread(sd["video_path"])
    else:
        first_frame_arr = iio.imread(sd["video_path"], index=0)
    # Match training first-frame preprocessing: the dataset runs every frame through
    # ImageCropAndResize (aspect-preserving scale-to-cover + center crop), NOT a
    # stretch resize. Use the same op so the rollout anchor stays in-distribution.
    first_pil = ImageCropAndResize(
        height=int(cfg.get("height", 480)),
        width=int(cfg.get("width", 832)),
        max_pixels=1920 * 1080,
        height_division_factor=16,
        width_division_factor=16,
    )(Image.fromarray(first_frame_arr))
    _to_gpu(pipe.vae, device)
    image = pipe.preprocess_image(first_pil).transpose(0, 1)  # [C, 1, H, W]
    first_frame_latent = pipe.vae.encode([image], device=device, tiled=False).to(torch.bfloat16)
    # Wan VAE returns [B, C, 1, h, w]; permute to CF API [B, 1, C, h, w]
    first_frame_latent = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
    _to_cpu(pipe.vae)

    # 5) KV-cache rollout — swap DiT to GPU; this is the heaviest phase.
    _to_gpu(pipe.dit, device)
    target_latent_frames = int(args.latent_frames) if args.latent_frames is not None \
        else int(cfg.get("sanity_sample_latent_frames", 300))
    diffusion_steps = int(args.diffusion_steps) if args.diffusion_steps is not None \
        else int(cfg.get("sanity_sample_diffusion_steps", 50))
    sampler = (args.sampler or cfg.get("sanity_sample_sampler", "ode")).lower()
    if sampler not in {"ode", "dmd"}:
        raise ValueError(f"Unsupported sanity_sample_sampler={sampler!r}; expected 'ode' or 'dmd'")
    action_per_frame = sd["action"].to(device).to(torch.bfloat16)
    print(
        f"[Sanity] rollout: target_latent={target_latent_frames}, "
        f"diffusion_steps={diffusion_steps}, sampler={sampler}, "
        f"action_T={action_per_frame.shape[1]}, "
        f"cfg={'on(g=%.1f)' % guidance_scale if context_neg is not None else 'off'}",
        flush=True,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[Sanity] fixed seed={args.seed}", flush=True)

    sample_fn = sample_dmd_kv_cache if sampler == "dmd" else sample_multistep_kv_cache
    output_latents = sample_fn(
        model=pipe.dit,
        action_per_frame=action_per_frame,
        context=context,
        context_neg=context_neg,
        guidance_scale=guidance_scale,
        first_frame_latent=first_frame_latent,
        target_latent_frames=target_latent_frames,
        diffusion_steps=diffusion_steps,
        kv_window_size=int(cfg.get("sanity_sample_kv_window_size", 16)),
        sink_size=int(cfg.get("sanity_sample_sink_size", 2)),
        num_frame_per_block=int(cfg.get("num_frame_per_block", 1)),
        scheduler_shift=float(cfg.get("timestep_shift", 5.0)),
        device=device,
        dtype=torch.bfloat16,
    )  # [1, F_lat, C, H, W]
    _to_cpu(pipe.dit)

    # 6) VAE decode + mp4 write — swap VAE back to GPU for decode.
    latent_wan = output_latents.permute(0, 2, 1, 3, 4).contiguous()  # [1, C, F, H, W]
    _to_gpu(pipe.vae, device)
    video = pipe.vae.decode(latent_wan, device=device, tiled=False)
    frames = pipe.vae_output_to_video(video)  # list of PIL.Image
    _to_cpu(pipe.vae)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import numpy as np

    # Use imageio-ffmpeg (already installed) via libx264 — pyav is not bundled.
    # 强制最大兼容编码: yuv420p + baseline profile + faststart(moov 前置). imageio 默认
    # 输出 High profile 且 moov 在文件尾, 导致 VSCode 媒体扩展 / 网页播放器打不开.
    video_array = np.stack([np.array(f) for f in frames], axis=0)  # [T, H, W, C] uint8
    iio.imwrite(
        args.out, video_array, fps=args.output_fps, codec="libx264", plugin="FFMPEG",
        pixelformat="yuv420p",
        output_params=["-profile:v", "baseline", "-movflags", "+faststart"],
    )
    print(f"[Sanity] wrote {args.out} ({len(frames)} pixel frames, fps={args.output_fps})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
