"""Frame-folded per-frame text cross-attention for LTX BasicAVTransformerBlock.

install_frame_context(model, num_frames, ctx_len) monkey-patches every
BasicAVTransformerBlock instance so that when the video context length equals
num_frames * ctx_len, the text cross-attention runs FRAME-FOLDED:

    x:(B, F*tpf, D) -> (B*F, tpf, D)
    ctx:(B, F*L, D) -> (B*F, L, D)

i.e. frame k's tokens attend ONLY frame k's context slice. Everything else
(AdaLN inside _apply_text_cross_attention, attn weights, gates) is reused by
delegating to the original method on the folded tensors. Audio contexts and
ordinary global prompts (length != F*L) pass through unchanged.
"""
from __future__ import annotations

import types

import torch

from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

_ORIG = BasicAVTransformerBlock._apply_text_cross_attention


def _folded_apply(self, x_normed, context, attn, scale_shift_table,
                  prompt_scale_shift_table, timestep, prompt_timestep,
                  context_mask, cross_attention_adaln=False):
    F = self._fc_num_frames
    L = self._fc_ctx_len
    B, T, D = x_normed.shape
    if context.shape[1] == F * L and T % F == 0:
        tpf = T // F
        xf = x_normed.reshape(B * F, tpf, D)
        cf = context.reshape(B * F, L, context.shape[-1])
        # timestep: (B, T, rest...) -> (B*F, tpf, rest...)
        tf = timestep.reshape(B * F, tpf, *timestep.shape[2:])
        ptf = None
        if prompt_timestep is not None:
            ptf = prompt_timestep.repeat_interleave(F, dim=0)
        out = _ORIG(self, xf, cf, attn, scale_shift_table,
                    prompt_scale_shift_table, tf, ptf, None,
                    cross_attention_adaln)
        return out.reshape(B, T, D)
    return _ORIG(self, x_normed, context, attn, scale_shift_table,
                 prompt_scale_shift_table, timestep, prompt_timestep,
                 context_mask, cross_attention_adaln)


def install_frame_context(model: torch.nn.Module, num_frames: int, ctx_len: int) -> int:
    """Patch all blocks in `model`; returns number of blocks patched."""
    n = 0
    for m in model.modules():
        if isinstance(m, BasicAVTransformerBlock):
            m._fc_num_frames = num_frames
            m._fc_ctx_len = ctx_len
            m._apply_text_cross_attention = types.MethodType(_folded_apply, m)
            n += 1
    return n
