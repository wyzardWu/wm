"""ReactiveGWM inference core: pipe loader + single-clip run_inference.

Designed to be friendly to multiprocessing: each worker process calls
`build_pipe()` once after binding to a single CUDA device, then
`run_inference()` per clip. State stays per-process; no inter-process state.

The button schema is sourced from a `GameProfile` (see `data/profiles.py`),
which keeps this module game-agnostic — SF2 / SF3 / future titles share it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
_REACTIVE_GWM = _HERE.parent
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from data.action_utils import hold_last_upsample  # noqa: E402
from data.profiles import GameProfile  # noqa: E402

from diffsynth.models.reactive_gwm_dit import ReactiveGWMModel  # noqa: E402
from diffsynth.pipelines.reactive_gwm import (  # noqa: E402
    ModelConfig, ReactiveGWMPipeline,
)
from diffsynth.utils.data import VideoData, crop_and_resize  # noqa: E402


WAN_MODEL_KWARGS = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48, "dim": 3072, "ffn_dim": 14336, "freq_dim": 256,
    "text_dim": 4096, "out_dim": 48, "num_heads": 24, "num_layers": 30,
    "eps": 1e-06,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def transfer_weights(custom_model, pretrained_dit):
    pretrained = pretrained_dit.state_dict()
    custom = custom_model.state_dict()
    new = {k: v for k, v in pretrained.items()
           if k in custom and v.shape == custom[k].shape}
    custom_model.load_state_dict(new, strict=False)
    return custom_model


@dataclass
class CkptSpec:
    """Ckpt loading recipe.

      full_ckpt: path to a full-DiT state dict (from full / scoped training).
      lora_ckpt: path to a LoRA state dict (from LoRA training).
      lora_alpha: LoRA scaling factor at inference time.
      Either full_ckpt or lora_ckpt may be None; both None -> base + cold-start ActionModule.
    """
    full_ckpt: Optional[str] = None
    lora_ckpt: Optional[str] = None
    lora_alpha: float = 0.8


def build_pipe(profile: GameProfile, base_dir: str, ckpt: CkptSpec,
               device: str = "cuda") -> ReactiveGWMPipeline:
    """Construct the inference pipeline. Call once per process after CUDA binding."""
    pipe = ReactiveGWMPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[
            ModelConfig(local_model_path=base_dir, model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        download_source="huggingface"),
            ModelConfig(local_model_path=base_dir, model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        download_source="huggingface"),
            ModelConfig(local_model_path=base_dir, model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="Wan2.2_VAE.pth",
                        download_source="huggingface"),
        ],
        tokenizer_config=ModelConfig(
            local_model_path=base_dir, model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/", download_source="huggingface",
        ),
        redirect_common_files=False,
    )
    custom_dit = ReactiveGWMModel(num_buttons=profile.num_buttons,
                                  **WAN_MODEL_KWARGS).to(torch.bfloat16)
    pipe.dit = transfer_weights(custom_dit, pipe.dit).to(pipe.device)
    del custom_dit
    torch.cuda.empty_cache()

    if ckpt.full_ckpt:
        state = load_file(ckpt.full_ckpt)
        missing, unexpected = pipe.dit.load_state_dict(state, strict=False)
        print(f"[ckpt] full: {len(state)} keys "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
        del state
        torch.cuda.empty_cache()
    if ckpt.lora_ckpt:
        print(f"[ckpt] lora: alpha={ckpt.lora_alpha} <- {ckpt.lora_ckpt}")
        pipe.load_lora(pipe.dit, ckpt.lora_ckpt, alpha=ckpt.lora_alpha)
    return pipe


def load_action_from_parquet(profile: GameProfile, path: str, num_frames: int,
                             device: str = "cuda",
                             hold_window: int | None = None) -> torch.Tensor:
    """parquet -> [1, T, num_buttons] bf16 keyboard action tensor on `device`.

    Parquet columns must be in `profile.button_cols` order. The 10Hz->20fps
    densification is handled by hold_last_upsample (window from profile by default).
    """
    win = profile.default_action_hold_window if hold_window is None else hold_window
    df = pd.read_parquet(path)
    arr = df[list(profile.button_cols)].values[:num_frames].astype(np.float32)
    keyboard = torch.tensor(arr)
    keyboard = hold_last_upsample(keyboard, window=win)
    return keyboard.unsqueeze(0).to(device, torch.bfloat16)


def make_constant_action(profile: GameProfile, num_frames: int,
                         button_idx_list: list[int],
                         device: str = "cuda") -> torch.Tensor:
    """Build a constant-pressed action tensor (e.g. always LEFT). Useful for
    eval_action.py where we want to feed a fixed button across all frames."""
    arr = np.zeros((num_frames, profile.num_buttons), dtype=np.float32)
    for b in button_idx_list:
        arr[:, b] = 1.0
    return torch.tensor(arr).unsqueeze(0).to(device, torch.bfloat16)


def first_frame_pil(video_path: str, height: int, width: int) -> Image.Image:
    return crop_and_resize(VideoData(video_path, height=height, width=width)[0],
                           height, width)


def run_inference(
    pipe: ReactiveGWMPipeline,
    prompt: str,
    keyboard_action: torch.Tensor,
    input_image: Image.Image,
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int = 30,
    cfg_scale: float = 1.0,
    action_cfg_scale: float = 1.0,
    seed: int = 0,
):
    """Run a single inference pass. Returns the pipe's native PIL frame list."""
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        input_image=input_image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        action_cfg_scale=action_cfg_scale,
        height=height, width=width,
        seed=seed, tiled=True,
        keyboard_action=keyboard_action,
    )
