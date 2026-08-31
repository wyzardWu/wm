"""CausalForcingReactiveGWMModel — Stage 1/2/3 共用因果 DiT.

继承自 `diffsynth.models.reactive_gwm_dit.ReactiveGWMModel`. 内部 forward 提供
三路径 dispatch:
  - `clean_x is not None, kv_cache is None`  → TF dual-block (Stage 1/2 训练)
  - `clean_x is None, kv_cache is not None`  → KV-cache 自回归 (Stage 1+ sanity / Stage 3 rollout / 最终推理)
  - 都为 None                                  → plain block-causal (verify/debug)

移植自 https://github.com/thu-ml/Causal-Forcing/blob/main/wan/modules/causal_model.py
但适配 ReactiveGWMModel 的命名 (DiTBlock 内含 self_attn/cross_attn/ffn/modulation)
与 per-block action_embedders. 与 SF 项目同名文件并列存在但**互不引用**.

State dict: 与父类 ReactiveGWMModel / 双向 base 100% 兼容 (不新增 nn.Parameter).
KV-cache 是临时 dict, 非 Module 属性, 不进 state dict.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from diffsynth.models.reactive_gwm_dit import (
    AttentionModule,
    ReactiveGWMModel,
    flash_attention,
    modulate,
    rope_apply,
    sinusoidal_embedding_1d,
)


# torch.compile flex_attention once (PyTorch 2.4+). Fallback to eager if compile fails.
try:
    _flex_attention = torch.compile(flex_attention, dynamic=False, mode="default")
except Exception:  # noqa: BLE001
    _flex_attention = flex_attention


class CausalForcingReactiveGWMModel(ReactiveGWMModel):
    """ReactiveGWM with causal block-attention + dual-block TF + KV-cache rollout."""

    def __init__(
        self,
        kv_window_size: int = 16,
        sink_size: int = 2,
        num_frame_per_block: int = 1,
        text_len: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kv_window_size = kv_window_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.text_len = text_len
        # flex_attention BlockMask cache: dict[(F_lat, frame_seqlen, npfb, device-str)] -> BlockMask
        self._tf_mask_cache: dict[tuple, BlockMask] = {}
        self._bc_mask_cache: dict[tuple, BlockMask] = {}

    # ================================================================== state-dict safety
    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        # Strip the runtime-only mask caches (they are torch BlockMask objects,
        # not nn.Parameter / buffers).
        return super().state_dict(*args, **kwargs)

    # ================================================================== TF mask (移植 CF 上游)
    @staticmethod
    def _build_teacher_forcing_mask(
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
    ) -> BlockMask:
        """Dual-block `[clean ‖ noisy]` BlockMask (CF 上游 `_prepare_teacher_forcing_mask`).

        Layout (total length = `2·num_frames·frame_seqlen`):
          - clean→clean: block-causal — `q_clean_i` sees `k_clean_{≤end-of-block-i}`
          - noisy→clean: `q_noisy_i` sees `k_clean` in strictly previous blocks
          - noisy→noisy: only within the same block
          - clean→noisy: blocked
          - identity (q==kv) always allowed
        """
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        attn_block_size = frame_seqlen * num_frame_per_block

        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Clean half: block-causal
        for start in range(0, num_frames * frame_seqlen, attn_block_size):
            context_ends[start : start + attn_block_size] = start + attn_block_size

        # Noisy half: each block sees [prev clean blocks] ∪ [same noisy block]
        for block_idx, n_start in enumerate(
            range(num_frames * frame_seqlen, total_length, attn_block_size)
        ):
            n_end = n_start + attn_block_size
            noise_noise_starts[n_start:n_end] = n_start
            noise_noise_ends[n_start:n_end] = n_end
            noise_context_ends[n_start:n_end] = block_idx * attn_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            c1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            c2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (c1 | c2)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

    @staticmethod
    def _build_blockwise_causal_mask(
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
    ) -> BlockMask:
        """单序列 block-causal mask (plain branch; verify 用)."""
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        attn_block_size = frame_seqlen * num_frame_per_block
        for start in range(0, total_length, attn_block_size):
            ends[start : start + attn_block_size] = start + attn_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

        return create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

    def _get_tf_mask(self, F_lat: int, frame_seqlen: int, npfb: int, device: torch.device) -> BlockMask:
        key = (F_lat, frame_seqlen, npfb, str(device))
        if key not in self._tf_mask_cache:
            self._tf_mask_cache[key] = self._build_teacher_forcing_mask(
                device, F_lat, frame_seqlen, npfb
            )
        return self._tf_mask_cache[key]

    def _get_bc_mask(self, F_lat: int, frame_seqlen: int, npfb: int, device: torch.device) -> BlockMask:
        key = (F_lat, frame_seqlen, npfb, str(device))
        if key not in self._bc_mask_cache:
            self._bc_mask_cache[key] = self._build_blockwise_causal_mask(
                device, F_lat, frame_seqlen, npfb
            )
        return self._bc_mask_cache[key]

    # ================================================================== KV-cache init / refill / update
    def init_kv_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        frame_seqlen: int,
        kv_window_size: int | None = None,
        sink_size: int | None = None,
    ) -> dict:
        """Allocate per-layer self-attn + cross-attn cache.

        Self-attn cache size = `(sink_size + kv_window_size) * frame_seqlen` tokens
        (大于此长度的历史会被 sink+recent 驱逐).

        Cross-attn cache 大小固定 = `text_len` tokens; 首次 forward 时填充, 后续帧复用.
        """
        if kv_window_size is None:
            kv_window_size = self.kv_window_size
        if sink_size is None:
            sink_size = self.sink_size
        num_heads = self.blocks[0].self_attn.num_heads
        head_dim = self.dim // num_heads
        self_cache_size = (sink_size + kv_window_size) * frame_seqlen

        self_cache = []
        cross_cache = []
        for _ in range(len(self.blocks)):
            self_cache.append(
                {
                    "k": torch.zeros(batch_size, self_cache_size, num_heads, head_dim, dtype=dtype, device=device),
                    "v": torch.zeros(batch_size, self_cache_size, num_heads, head_dim, dtype=dtype, device=device),
                    "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
                    "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
                    "frame_seqlen": frame_seqlen,
                    "kv_window_size": kv_window_size,
                    "sink_size": sink_size,
                }
            )
            cross_cache.append(
                {
                    "k": torch.zeros(batch_size, self.text_len, num_heads, head_dim, dtype=dtype, device=device),
                    "v": torch.zeros(batch_size, self.text_len, num_heads, head_dim, dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        return {"self": self_cache, "cross": cross_cache, "frame_seqlen": frame_seqlen}

    @staticmethod
    def reset_kv_cache(kv_cache: dict) -> None:
        for c in kv_cache["self"]:
            c["global_end_index"].fill_(0)
            c["local_end_index"].fill_(0)
        for c in kv_cache["cross"]:
            c["is_init"] = False

    # ================================================================== forward dispatcher
    def forward(  # type: ignore[override]
        self,
        noisy_x: torch.Tensor,
        timestep: torch.Tensor,
        *,
        clean_x: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        action: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        first_frame_anchor: bool = False,
        write_to_cache: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Three-way dispatch.

        Shapes:
          noisy_x: TF       `[B, F, C, H, W]`
                   KV-cache `[B, F_block, C, H, W]` (F_block usually = num_frame_per_block)
                   plain    `[B, F, C, H, W]`
          clean_x: same as noisy_x in TF, else None.
          timestep: per-frame `[B, F]` (or `[B, F_block]`).
          action: `[B, T_pixel, num_buttons]` raw keyboard sequence, **None** allowed.
          context: T5 embedding `[B, L_text, text_dim]`.
          write_to_cache: only used by KV-cache path. `False` for multi-step
            diffusion intermediate steps (current K/V are scratch); `True` for
            the final clean refill that commits this frame to history.
        """
        if clean_x is not None and kv_cache is None:
            return self._forward_tf(
                noisy_x=noisy_x,
                clean_x=clean_x,
                timestep_per_frame=timestep,
                action=action,
                context=context,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                first_frame_anchor=first_frame_anchor,
            )
        if clean_x is None and kv_cache is not None:
            return self._forward_kv_cache(
                noisy_x=noisy_x,
                timestep_per_frame=timestep,
                action=action,
                context=context,
                kv_cache=kv_cache,
                write_to_cache=write_to_cache,
            )
        return self._forward_plain(
            noisy_x=noisy_x,
            timestep_per_frame=timestep,
            action=action,
            context=context,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    # ================================================================== helpers shared by 3 paths
    def _patchify_and_geom(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """`[B, F, C, H, W]` → tokens `[B, F·h·w, dim]`, return (tokens, F, h, w).

        VAE latent: F = num_latent_frames (post 4× temporal compression).
        patch_size (1,2,2) halves H and W.
        """
        # Convert to [B, C, F, H, W] for Conv3d patch_embedding
        x_cfhw = x.permute(0, 2, 1, 3, 4).contiguous()
        x_pe = self.patch_embedding(x_cfhw)  # [B, dim, F, h, w]
        F_lat, h, w = x_pe.shape[2], x_pe.shape[3], x_pe.shape[4]
        tokens = rearrange(x_pe, "b c f h w -> b (f h w) c").contiguous()
        return tokens, F_lat, h, w

    def _unpatchify(self, tokens: torch.Tensor, F_lat: int, h: int, w: int) -> torch.Tensor:
        """`[B, F·h·w, dim]` → `[B, F, C_out, H, W]` (CF API layout)."""
        x_cfhw = rearrange(
            tokens,
            "b (f h w) (px py pz c) -> b c (f px) (h py) (w pz)",
            f=F_lat,
            h=h,
            w=w,
            px=self.patch_size[0],
            py=self.patch_size[1],
            pz=self.patch_size[2],
        )  # [B, C, F, H, W]
        return x_cfhw.permute(0, 2, 1, 3, 4).contiguous()

    def _build_rope_freqs(self, F_lat: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """3D RoPE freqs (与 WanModel.forward 一致)."""
        freqs = torch.cat(
            [
                self.freqs[0][:F_lat].view(F_lat, 1, 1, -1).expand(F_lat, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(F_lat, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(F_lat, h, w, -1),
            ],
            dim=-1,
        ).reshape(F_lat * h * w, 1, -1)
        return freqs.to(device)

    def _per_token_t_mod(
        self,
        timestep_per_frame: torch.Tensor,
        F_lat: int,
        spatial_per_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`[B, F]` per-frame timestep → `t [B, F·spatial, dim]`, `t_mod [B, F·spatial, 6, dim]`.

        Broadcasts each frame's timestep across its `spatial_per_frame` tokens.
        """
        B = timestep_per_frame.shape[0]
        # Repeat each frame's t across its spatial tokens → [B, F·spatial]
        ts = timestep_per_frame.repeat_interleave(spatial_per_frame, dim=1)  # [B, F·s]
        # sinusoidal_embedding_1d expects 1D position vector → reshape to [B*F*s]
        emb = sinusoidal_embedding_1d(self.freq_dim, ts.reshape(-1)).reshape(B, F_lat * spatial_per_frame, self.freq_dim)
        t = self.time_embedding(emb.to(dtype=self.patch_embedding.weight.dtype))
        t_mod = self.time_projection(t).unflatten(2, (6, self.dim))
        return t, t_mod

    # ================================================================== self-attn variants
    @staticmethod
    def _sa_qkv(sa, x: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute rope-applied Q/K and raw V from SelfAttention(sa) for input x."""
        q = sa.norm_q(sa.q(x))
        k = sa.norm_k(sa.k(x))
        v = sa.v(x)
        q = rope_apply(q, freqs, sa.num_heads)
        k = rope_apply(k, freqs, sa.num_heads)
        return q, k, v

    def _self_attn_flex(
        self, sa, x: torch.Tensor, freqs: torch.Tensor, block_mask: BlockMask
    ) -> torch.Tensor:
        """Self-attention with flex_attention + BlockMask (TF or plain block-causal)."""
        q, k, v = self._sa_qkv(sa, x, freqs)
        # q/k/v: [B, L, dim]; reshape to [B, num_heads, L_pad, head_dim]; pad to ×128
        B, L, dim = q.shape
        nh = sa.num_heads
        hd = dim // nh
        q_h = q.reshape(B, L, nh, hd).transpose(1, 2)  # [B, nh, L, hd]
        k_h = k.reshape(B, L, nh, hd).transpose(1, 2)
        v_h = v.reshape(B, L, nh, hd).transpose(1, 2)
        # Pad L to multiple of 128 (flex_attention requirement)
        pad = math.ceil(L / 128) * 128 - L
        if pad > 0:
            zero_pad = lambda t: F.pad(t, (0, 0, 0, pad))
            q_h = zero_pad(q_h)
            k_h = zero_pad(k_h)
            v_h = zero_pad(v_h)
        out = _flex_attention(q_h, k_h, v_h, block_mask=block_mask)
        if pad > 0:
            out = out[:, :, :L, :]
        out = out.transpose(1, 2).reshape(B, L, dim)
        return sa.o(out)

    def _self_attn_kv_cache(
        self,
        sa,
        x: torch.Tensor,
        freqs_block: torch.Tensor,
        kv_cache_layer: dict,
        current_start: int,
        write_to_cache: bool = True,
    ) -> torch.Tensor:
        """Self-attention for a single block at `current_start`, using KV-cache.

        Two modes, distinguished by `write_to_cache`:
          - True (refill): write current K/V to cache, evict if full, advance
            global/local_end. Used by `refill_kv_cache` to commit a clean frame.
          - False (multi-step diffusion intermediate): treat current K/V as
            scratch — attend over [historical cached K/V | current K/V] but do
            NOT modify cache or advance indices. Without this, every diffusion
            step on the same frame would burn one more cache slot + advance
            global_end, so the rope `start_frame = global_end // frame_seqlen`
            quickly exceeds the precomputed `end=1024`, returning empty
            f_freqs and breaking attention with a shape mismatch.
        """
        q, k, v = self._sa_qkv(sa, x, freqs_block)
        B, num_new, dim = q.shape
        nh = sa.num_heads
        hd = dim // nh

        cache_k = kv_cache_layer["k"]
        cache_v = kv_cache_layer["v"]
        local_end = int(kv_cache_layer["local_end_index"].item())
        global_end = int(kv_cache_layer["global_end_index"].item())
        cache_size = cache_k.shape[1]
        sink_tokens = kv_cache_layer["sink_size"] * kv_cache_layer["frame_seqlen"]
        max_attn_size = kv_cache_layer["kv_window_size"] * kv_cache_layer["frame_seqlen"] + sink_tokens

        if not write_to_cache:
            # Read historical cached K/V up to current local_end (excludes the
            # frame we're denoising), then concat current frame's K/V to the
            # right for attention. Don't touch the cache.
            attn_start = max(0, local_end - max_attn_size)
            if local_end > 0:
                k_hist_flat = cache_k[:, attn_start:local_end].reshape(B, -1, dim)
                v_hist_flat = cache_v[:, attn_start:local_end].reshape(B, -1, dim)
                k_full = torch.cat([k_hist_flat, k], dim=1)
                v_full = torch.cat([v_hist_flat, v], dim=1)
            else:
                k_full = k
                v_full = v
            out = flash_attention(q, k_full, v_full, num_heads=nh)
            return sa.o(out)

        # write_to_cache=True path: original behaviour.
        current_end = current_start + num_new

        k_bnhd = k.reshape(B, num_new, nh, hd)
        v_bnhd = v.reshape(B, num_new, nh, hd)

        if current_end > global_end and num_new + local_end > cache_size:
            num_evicted = num_new + local_end - cache_size
            num_rolled = local_end - num_evicted - sink_tokens
            cache_k[:, sink_tokens : sink_tokens + num_rolled] = cache_k[
                :, sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled
            ].clone()
            cache_v[:, sink_tokens : sink_tokens + num_rolled] = cache_v[
                :, sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled
            ].clone()
            local_end = local_end + (current_end - global_end) - num_evicted
        elif current_end > global_end:
            local_end = local_end + (current_end - global_end)
        # else (re-write at same position): local_end unchanged

        local_start = local_end - num_new
        cache_k[:, local_start:local_end] = k_bnhd
        cache_v[:, local_start:local_end] = v_bnhd

        attn_start = max(0, local_end - max_attn_size)
        k_read = cache_k[:, attn_start:local_end]
        v_read = cache_v[:, attn_start:local_end]
        k_read_flat = k_read.reshape(B, -1, dim)
        v_read_flat = v_read.reshape(B, -1, dim)
        out = flash_attention(q, k_read_flat, v_read_flat, num_heads=nh)

        kv_cache_layer["global_end_index"].fill_(current_end)
        kv_cache_layer["local_end_index"].fill_(local_end)
        return sa.o(out)

    def _cross_attn_with_cache(
        self,
        ca,
        x: torch.Tensor,
        context: torch.Tensor,
        cross_cache_layer: dict | None,
    ) -> torch.Tensor:
        """Cross-attention; reuses cached K/V across all frames in a rollout."""
        q = ca.norm_q(ca.q(x))
        nh = ca.num_heads
        hd = ca.dim // nh
        B = x.shape[0]
        if cross_cache_layer is None:
            # No cache → regular cross-attn
            k = ca.norm_k(ca.k(context))
            v = ca.v(context)
            out = flash_attention(q, k, v, num_heads=nh)
            return ca.o(out)
        if not cross_cache_layer["is_init"]:
            k = ca.norm_k(ca.k(context))
            v = ca.v(context)
            L_ctx = context.shape[1]
            cross_cache_layer["k"][:, :L_ctx] = k.reshape(B, L_ctx, nh, hd)
            cross_cache_layer["v"][:, :L_ctx] = v.reshape(B, L_ctx, nh, hd)
            cross_cache_layer["is_init"] = True
            cross_cache_layer["L_ctx"] = L_ctx
        L_ctx = cross_cache_layer.get("L_ctx", cross_cache_layer["k"].shape[1])
        k_flat = cross_cache_layer["k"][:, :L_ctx].reshape(B, L_ctx, ca.dim)
        v_flat = cross_cache_layer["v"][:, :L_ctx].reshape(B, L_ctx, ca.dim)
        out = flash_attention(q, k_flat, v_flat, num_heads=nh)
        return ca.o(out)

    # ================================================================== per-block forward
    def _block_forward(
        self,
        block,
        block_idx: int,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        action_binned: torch.Tensor | None,
        f: int,
        h: int,
        w: int,
        *,
        block_mask: BlockMask | None = None,
        kv_cache_layer: dict | None = None,
        cross_cache_layer: dict | None = None,
        current_start: int = 0,
        write_to_cache: bool = True,
    ) -> torch.Tensor:
        """Single DiTBlock forward, with action graft + selected self-attn path."""
        # Action graft (ReactiveGWMModel v4 per-block embedder)
        if action_binned is not None:
            x = self.inject_at_block(x, action_binned, block_idx, f, h, w)

        # Same modulation logic as DiTBlock.forward, but t_mod is [B, L_seq, 6, dim]
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)

        # Self-attention
        input_x = modulate(block.norm1(x), shift_msa, scale_msa)
        if kv_cache_layer is not None:
            sa_out = self._self_attn_kv_cache(
                block.self_attn, input_x, freqs, kv_cache_layer, current_start, write_to_cache,
            )
        elif block_mask is not None:
            sa_out = self._self_attn_flex(block.self_attn, input_x, freqs, block_mask)
        else:
            # Should not happen in dispatched forward paths
            sa_out = block.self_attn(input_x, freqs)
        x = block.gate(x, gate_msa, sa_out)

        # Cross-attention (text)
        x = x + self._cross_attn_with_cache(block.cross_attn, block.norm3(x), context, cross_cache_layer)

        # FFN
        input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(input_x))
        return x

    # ================================================================== TF dual-block (training)
    def _forward_tf(
        self,
        noisy_x: torch.Tensor,
        clean_x: torch.Tensor,
        timestep_per_frame: torch.Tensor,
        *,
        action: torch.Tensor | None,
        context: torch.Tensor,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
        first_frame_anchor: bool,
    ) -> torch.Tensor:
        """TF dual-block forward; returns noisy-half flow prediction [B, F, C, H, W].

        Frame 0 anchor: if `first_frame_anchor=True`, copies clean frame 0 into
        noisy frame 0 (callers usually already do this, but we enforce it).
        """
        B, F_in, C, H, W = noisy_x.shape
        if first_frame_anchor:
            noisy_x = noisy_x.clone()
            noisy_x[:, 0] = clean_x[:, 0]

        # Patchify both halves; same spatial geom.
        clean_tok, F_lat, h, w = self._patchify_and_geom(clean_x)
        noisy_tok, _, _, _ = self._patchify_and_geom(noisy_x)
        spatial = h * w  # tokens per frame
        L_half = F_lat * spatial

        # Concat: [B, 2·L_half, dim]
        x = torch.cat([clean_tok, noisy_tok], dim=1)

        # Per-frame timestep tensor for both halves
        # clean half: t=0 (all frames). noisy half: timestep_per_frame from caller.
        # Exception (zhiyang-noising): when the caller passes clean_x that IS the
        # noisy sequence (plain-causal semantics on the dual-block path), it sets
        # self._context_half_true_t so the context half is conditioned on the
        # true timestep instead of a bogus t=0.
        if getattr(self, "_context_half_true_t", False):
            ts_clean = timestep_per_frame
        else:
            ts_clean = torch.zeros_like(timestep_per_frame)
        ts_dual = torch.cat([ts_clean, timestep_per_frame], dim=1)  # [B, 2F]
        _t_emb, t_mod = self._per_token_t_mod(ts_dual, 2 * F_lat, spatial)

        # 3D RoPE for the doubled sequence — clean and noisy share the same
        # per-frame indices [0..F-1], because the TF mask handles visibility.
        freqs_half = self._build_rope_freqs(F_lat, h, w, x.device)
        freqs = torch.cat([freqs_half, freqs_half], dim=0)  # [2·L_half, 1, dim]

        # Text embedding
        ctx = self.text_embedding(context)

        # Action binning. The base inject_at_block expects action_binned length
        # to match f frames; here x is dual-block (clean ‖ noisy) so we duplicate
        # the per-frame action so BOTH halves receive the same per-frame action
        # signal — without this, clean half goes through the inject_at_block
        # `prefix-padding` branch and gets zero action bias, breaking train /
        # KV-cache-inference parity (KV-cache rollout injects action on every
        # frame including the cached history blocks).
        action_binned = (
            self.prepare_action_binned(action, F_lat) if action is not None else None
        )
        action_binned_dual = (
            torch.cat([action_binned, action_binned], dim=1) if action_binned is not None else None
        )
        f_dual = 2 * F_lat

        # BlockMask
        block_mask = self._get_tf_mask(F_lat, spatial, self.num_frame_per_block, x.device)

        # Block loop
        for block_idx, block in enumerate(self.blocks):
            if self.training and use_gradient_checkpointing:
                from ..core.gradient import gradient_checkpoint_forward

                x = gradient_checkpoint_forward(
                    self._block_forward,
                    True,
                    use_gradient_checkpointing_offload,
                    block,
                    block_idx,
                    x,
                    ctx,
                    t_mod,
                    freqs,
                    action_binned_dual,
                    f_dual,
                    h,
                    w,
                    block_mask=block_mask,
                )
            else:
                x = self._block_forward(
                    block,
                    block_idx,
                    x,
                    ctx,
                    t_mod,
                    freqs,
                    action_binned_dual,
                    f_dual,
                    h,
                    w,
                    block_mask=block_mask,
                )

        # Final head — t for head is the per-token time embedding (already computed
        # via the same `_per_token_t_mod` call above as `_t_emb`).
        x = self.head(x, _t_emb)

        # Slice noisy half and unpatchify
        x_noisy = x[:, L_half:]
        return self._unpatchify(x_noisy, F_lat, h, w)

    # ================================================================== KV-cache rollout (sanity / inference)
    def _forward_kv_cache(
        self,
        noisy_x: torch.Tensor,
        timestep_per_frame: torch.Tensor,
        *,
        action: torch.Tensor | None,
        context: torch.Tensor,
        kv_cache: dict,
        write_to_cache: bool = True,
    ) -> torch.Tensor:
        """Single-block forward over KV-cache; `noisy_x: [B, F_block, C, H, W]`.

        `write_to_cache=False` is the right choice for the multi-step diffusion
        intermediate forwards: K/V for the current frame are scratch, only the
        final clean `refill_kv_cache` should advance the cache pointers.
        """
        B, F_in = noisy_x.shape[:2]
        tokens, F_lat, h, w = self._patchify_and_geom(noisy_x)
        spatial = h * w
        L_block = F_lat * spatial

        # Per-token timestep
        _t, t_mod = self._per_token_t_mod(timestep_per_frame, F_lat, spatial)

        # Action binning (only for this block's F_lat frames)
        action_binned = (
            self.prepare_action_binned(action, F_lat) if action is not None else None
        )

        # Context (lazy text embedding — happens once via cross-attn cache fill)
        ctx = self.text_embedding(context)

        # RoPE freqs offset by global frame index in cache.
        # global_end advances ONLY on refill (write_to_cache=True), so during
        # multi-step diffusion start_frame stays pinned at the current frame.
        global_frame_end = int(kv_cache["self"][0]["global_end_index"].item())
        start_frame = global_frame_end // spatial
        # Take F_lat frames worth of freqs starting from start_frame
        f_freqs = self.freqs[0][start_frame : start_frame + F_lat].view(F_lat, 1, 1, -1).expand(F_lat, h, w, -1)
        h_freqs = self.freqs[1][:h].view(1, h, 1, -1).expand(F_lat, h, w, -1)
        w_freqs = self.freqs[2][:w].view(1, 1, w, -1).expand(F_lat, h, w, -1)
        freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).reshape(L_block, 1, -1).to(tokens.device)

        current_start = global_frame_end  # in token units

        x = tokens
        for block_idx, block in enumerate(self.blocks):
            x = self._block_forward(
                block,
                block_idx,
                x,
                ctx,
                t_mod,
                freqs,
                action_binned,
                F_lat,
                h,
                w,
                kv_cache_layer=kv_cache["self"][block_idx],
                cross_cache_layer=kv_cache["cross"][block_idx],
                current_start=current_start,
                write_to_cache=write_to_cache,
            )

        # Final head with the per-token t embedding (reuse `_t` from above)
        x = self.head(x, _t)
        return self._unpatchify(x, F_lat, h, w)

    def refill_kv_cache(
        self,
        x_block: torch.Tensor,
        action: torch.Tensor | None,
        context: torch.Tensor,
        kv_cache: dict,
    ) -> None:
        """Cache-only refill: run forward with t=0 to overwrite K/V with clean-context K/V.

        Output is discarded; what we need is the side-effect of writing K/V to
        `kv_cache` at the current frame slot. Mirrors CF 上游 Step 3.3 in
        `pipeline/causal_diffusion_inference.py`.
        """
        B, F_block = x_block.shape[:2]
        zero_t = torch.zeros(B, F_block, device=x_block.device, dtype=x_block.dtype)
        with torch.no_grad():
            _ = self._forward_kv_cache(
                noisy_x=x_block,
                timestep_per_frame=zero_t,
                action=action,
                context=context,
                kv_cache=kv_cache,
            )

    # ================================================================== plain block-causal (verify)
    def _forward_plain(
        self,
        noisy_x: torch.Tensor,
        timestep_per_frame: torch.Tensor,
        *,
        action: torch.Tensor | None,
        context: torch.Tensor,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> torch.Tensor:
        """Single-sequence block-causal forward (no KV-cache, no TF). Used for verify/debug."""
        tokens, F_lat, h, w = self._patchify_and_geom(noisy_x)
        spatial = h * w
        _t, t_mod = self._per_token_t_mod(timestep_per_frame, F_lat, spatial)
        ctx = self.text_embedding(context)
        action_binned = (
            self.prepare_action_binned(action, F_lat) if action is not None else None
        )
        freqs = self._build_rope_freqs(F_lat, h, w, tokens.device)
        block_mask = self._get_bc_mask(F_lat, spatial, self.num_frame_per_block, tokens.device)
        x = tokens
        for block_idx, block in enumerate(self.blocks):
            x = self._block_forward(
                block,
                block_idx,
                x,
                ctx,
                t_mod,
                freqs,
                action_binned,
                F_lat,
                h,
                w,
                block_mask=block_mask,
            )
        x = self.head(x, _t)
        return self._unpatchify(x, F_lat, h, w)
