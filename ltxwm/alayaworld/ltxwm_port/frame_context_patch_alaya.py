"""Frame-folded per-frame text cross-attention for AlayaWorld's LTX23AttentionBlock.

Port of ltxwm/frame_context_patch.py (official ltx-core seam) onto the
alayaworld ltx2 stack. Same contract:

    install_frame_context(model, num_frames, ctx_len) patches every
    LTX23AttentionBlock so that when context length == num_frames * ctx_len,
    text cross-attention runs frame-folded:

        x:(B, F*tpf, D)   -> (B*F, tpf, D)
        ctx:(B, F*L, Dc)  -> (B*F, L, Dc)

    frame k's tokens attend ONLY frame k's context slice. AdaLN inside
    _apply_text_cross_attention (2.3 cross_attention_adaln path) is reused by
    delegating to the original method on folded tensors; timesteps fold with x.
    Contexts of any other length pass through unchanged (global prompts, audio).
"""
from __future__ import annotations

import sys
import types

import torch

sys.path.insert(0, "/data/yuzhewu/ltxwm/alayaworld")
from ltx2.modules.model_ltx_2_3 import LTX23AttentionBlock

_ORIG = LTX23AttentionBlock._apply_text_cross_attention


def _folded_apply(self, x, context, timesteps, prompt_timestep, context_mask):
    F = self._fc_num_frames
    L = self._fc_ctx_len
    B, T, D = x.shape
    if context is not None and context.shape[1] == F * L and T % F == 0:
        tpf = T // F
        xf = x.reshape(B * F, tpf, D)
        cf = context.reshape(B * F, L, context.shape[-1])
        tf = timesteps.reshape(B * F, tpf, *timesteps.shape[2:])
        ptf = None
        if prompt_timestep is not None:
            ptf = prompt_timestep.repeat_interleave(F, dim=0)
        out = _ORIG(self, xf, cf, tf, ptf, None)
        return out.reshape(B, T, D)
    return _ORIG(self, x, context, timesteps, prompt_timestep, context_mask)


def install_frame_context(model: torch.nn.Module, num_frames: int, ctx_len: int) -> int:
    """Patch all LTX23AttentionBlock instances; returns count patched."""
    n = 0
    for m in model.modules():
        if isinstance(m, LTX23AttentionBlock):
            m._fc_num_frames = num_frames
            m._fc_ctx_len = ctx_len
            m._apply_text_cross_attention = types.MethodType(_folded_apply, m)
            n += 1
    return n
