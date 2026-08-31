"""
GameMaster DiT — a minimal, self-contained bidirectional Wan2.2 TI2V-5B DiT with
**per-frame per-entity** natural-language cross-attention.

Design provenance (copy-then-modify; originals are NOT touched):
  - Structure / weight names mirror DiffSynth-Studio's bidirectional WanModel
    (`DiffSynth-Studio/diffsynth/models/wan_video_dit.py`) so the real
    Wan2.2-TI2V-5B checkpoint can be load_state_dict'd later, AND so a checkpoint
    trained here transfers 1:1 to Incantation's CausalWanModel for the future
    causal-distillation stage.
  - The per-frame per-entity text conditioning is the Incantation increment
    (`Incantation/modules/causal_model.py`), realized for a PARALLEL training
    forward via handoff-doc correction C9: fold the frame dim into the batch dim
    for the cross-attention only; self-attention stays full-clip bidirectional.

What this is (v1):  bidirectional base (NOT causal, NOT streaming, NO distillation),
single timestep per clip (standard flow-matching SFT), per-frame DIFFERENT text.
What this is NOT (yet): causal mask, KV-cache, sliding window, per-frame timestep,
Self-Forcing distillation — all deferred to a later stage.

TI2V-5B config (confirmed): dim=3072, num_heads=24 (head_dim=128), num_layers=30,
ffn_dim=14336, in_dim=out_dim=48, text_dim=4096, freq_dim=256, patch=(1,2,2).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint   # explicit (not via optimizer side-effect)


# ───────────────────────── norms ─────────────────────────

class RMSNorm(nn.Module):
    """RMSNorm with learnable weight; computes in fp32 then casts back.
    Used as norm_q / norm_k on attention Q,K (matches Wan)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def layernorm(dim, affine, eps=1e-6):
    """Wan uses non-affine LayerNorm for norm1/norm2 and affine for norm3."""
    return nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)


# ───────────────────────── 3D RoPE ─────────────────────────
# Standard Wan formulation: head_dim is split across (temporal, H, W) axes and a
# complex rotation is applied per (f, h, w) grid position. Positions are absolute
# 0..F-1 / 0..H-1 / 0..W-1 (bidirectional; no causal offset). Frame-major token
# order: token index = f*(h*w) + i*w + j.

def _rope_freqs(max_pos, dim, theta=10000.0):
    """Per-axis complex freq table -> [max_pos, dim//2] complex."""
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    pos = torch.arange(max_pos).float()
    ang = torch.outer(pos, freqs)               # [max_pos, dim//2]
    return torch.polar(torch.ones_like(ang), ang)  # complex64


class Rope3D(nn.Module):
    """Precomputes the 3 axis tables; applies to q/k of shape [B, S, N, D]."""

    def __init__(self, head_dim, max_pos=1024, theta=10000.0):
        super().__init__()
        # split head_dim//2 complex cols across axes; temporal gets the remainder.
        c = head_dim // 2
        c_t = c - 2 * (c // 3)
        c_h = c // 3
        c_w = c // 3
        self.dims = (c_t, c_h, c_w)
        # NOTE: store as PLAIN attributes, NOT buffers. Module.to(dtype=...) would
        # cast complex buffers to real and silently destroy the rotation. These are
        # derived constants (not learned, not in state_dict); we move them to the
        # right device on demand inside forward.
        self.f_t = _rope_freqs(max_pos, c_t * 2, theta)
        self.f_h = _rope_freqs(max_pos, c_h * 2, theta)
        self.f_w = _rope_freqs(max_pos, c_w * 2, theta)

    def _grid_freqs(self, Fl, Hl, Wl, device, f_start=0):
        # build [Fl*Hl*Wl, head_dim//2] complex freqs in frame-major order.
        # f_start>0 places this chunk's frames at ABSOLUTE temporal positions
        # f_start..f_start+Fl-1 (the KV-cache path passes f_start=frame_index so a
        # single frame gets consecutive absolute position k, matching full-clip).
        ft = self.f_t[f_start:f_start + Fl].to(device)   # [Fl, c_t]
        fh = self.f_h[:Hl].to(device)           # [Hl, c_h]
        fw = self.f_w[:Wl].to(device)           # [Wl, c_w]
        ft = ft[:, None, None, :].expand(Fl, Hl, Wl, -1)
        fh = fh[None, :, None, :].expand(Fl, Hl, Wl, -1)
        fw = fw[None, None, :, :].expand(Fl, Hl, Wl, -1)
        grid = torch.cat([ft, fh, fw], dim=-1)  # [Fl,Hl,Wl, c]
        return grid.reshape(Fl * Hl * Wl, -1)   # [S, c] complex

    def forward(self, x, Fl, Hl, Wl, f_start=0):
        # x: [B, S, N, D] real. Apply complex rotation on last dim.
        B, S, N, D = x.shape
        freqs = self._grid_freqs(Fl, Hl, Wl, x.device, f_start=f_start)  # [S, D//2] complex
        xc = torch.view_as_complex(x.float().reshape(B, S, N, D // 2, 2))
        xc = xc * freqs[None, :, None, :]
        out = torch.view_as_real(xc).reshape(B, S, N, D)
        return out.to(x.dtype)


# ───────────────────────── attention ─────────────────────────

# Optional SageAttention (INT8-QK + FP8-PV) drop-in, gated by env GM_SAGEATTN=1. Default OFF =
# byte-identical original SDPA. Only used on the deploy rollout path: attn_mask is None AND
# no grad required (training's block-causal [S,S] mask and DMD's grad rollout fall back to SDPA).
import os as _os
_SAGE = None
if _os.environ.get("GM_SAGEATTN") == "1":
    try:
        from sageattention import sageattn as _SAGE
    except Exception:
        _SAGE = None

# X-Cache (training-free cross-frame block-residual reuse), gated by env GM_XCACHE=1. Default OFF =
# byte-identical original logic (init_xcache never called, forward_frame's step_index/xcache stay None).
# Only active on the per-frame streaming path (forward_frame) with commit=False (the transient denoise
# passes); commit=True passes are always fully computed so the KV cache stays exact.
_XCACHE = _os.environ.get("GM_XCACHE") == "1"
_XCACHE_TAU = float(_os.environ.get("GM_XCACHE_TAU", "0.97"))   # fixed cosine hit threshold (matches probe)


def _xcache_fp(h, Hl, Wl, G=4):
    """Compact block-input fingerprint IDENTICAL to scripts/xcache_probe.py: concat(global channel
    mean [dim], GxG spatial avg-pool [dim*G*G]). Cheap (no full-tensor cosine) and smooths local noise
    so cross-frame similarity is measured the same way the probe validated (~90% skippable)."""
    x = h.detach().float()                                    # [B, hw, dim]
    B, hw, D = x.shape
    gm = x.mean(dim=1).reshape(-1)                            # [B*D] global channel mean over tokens
    grid = x.transpose(1, 2).reshape(B, D, Hl, Wl)           # [B, D, Hl, Wl]
    sp = F.adaptive_avg_pool2d(grid, (G, G)).reshape(-1)      # [B*D*G*G]
    return torch.cat([gm, sp])


def sdpa(q, k, v, attn_mask=None):
    """q,k,v: [B, S, N, D] -> [B, S, N, D]. Bidirectional SDPA, or block-causal when
    a bool attn_mask [S,S] (True=attend) is given (the causal student; a [S,S] mask
    picks the mem-efficient backend, so the full score matrix is not materialized)."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # [B,N,S,D]
    if _SAGE is not None and attn_mask is None and not q.requires_grad:
        o = _SAGE(q, k, v, tensor_layout="HND", is_causal=False)   # [B,N,S,D]
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return o.transpose(1, 2)                            # [B,S,N,D]


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # Wan/Incantation QK-norm: RMSNorm over the FULL dim (all heads jointly),
        # applied to the flat [B,S,dim] projection BEFORE the per-head reshape.
        # (Real Wan2.2-TI2V-5B checkpoint has [dim]=[3072] norm weights; using
        # head_dim here would break load_state_dict AND change the RMS axis.)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)

    def forward(self, x, rope, Fl, Hl, Wl, attn_mask=None,
                kv_cache=None, layer_idx=None, frame_index=None, commit=False, tf=False):
        B, S, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(B, S, n, d)
        k = self.norm_k(self.k(x)).view(B, S, n, d)
        v = self.v(x).view(B, S, n, d)
        if kv_cache is None:
            # ── UNCHANGED bidir / full-clip-causal path ──
            if tf:
                # ── TEACHER-FORCING 2·F stream: the sequence is [clean F | noisy F]; RoPE EACH
                # half independently from f_start=0 (CF causal_model.py:124-133) so clean-frame f
                # and noisy-frame f share temporal RoPE position f (NOT roped 0..2Fl-1). ──
                qa, qb = q.chunk(2, dim=1); ka, kb = k.chunk(2, dim=1)
                q = torch.cat([rope(qa, Fl, Hl, Wl), rope(qb, Fl, Hl, Wl)], dim=1)
                k = torch.cat([rope(ka, Fl, Hl, Wl), rope(kb, Fl, Hl, Wl)], dim=1)
            else:
                q = rope(q, Fl, Hl, Wl)
                k = rope(k, Fl, Hl, Wl)
            o = sdpa(q, k, v, attn_mask=attn_mask).reshape(B, S, n * d)
            return self.o(o)
        if kv_cache.get("kv_window") is None:
            # ── absolute KV-cache path (UNCHANGED; training + proven equivalence test) ──
            # RoPE is absolute, so caching POST-RoPE K is exact (no re-rope on read). Frame k
            # attends ALL of frames 0..k (cache holds 0..k-1, plus self) -> no intra-set mask.
            q = rope(q, Fl, Hl, Wl, f_start=frame_index)
            k = rope(k, Fl, Hl, Wl, f_start=frame_index)
            k_hist, v_hist = kv_cache["k"][layer_idx], kv_cache["v"][layer_idx]
            if k_hist is not None:
                k_cat = torch.cat([k_hist, k], dim=1)    # [B, (frame_index+1)*hw, n, d]
                v_cat = torch.cat([v_hist, v], dim=1)
            else:
                k_cat, v_cat = k, v
            o = sdpa(q, k_cat, v_cat, attn_mask=None).reshape(B, S, n * d)
            if commit:                                   # persist ONLY on the clean/final pass
                kv_cache["k"][layer_idx] = k_cat.detach()
                kv_cache["v"][layer_idx] = v_cat.detach()
            return self.o(o)
        # ── sliding-window KV-cache path (bounded long rollout): cache RAW K, re-RoPE on read
        # at BOUNDED local positions so RoPE never goes OOD. Buffer = [sink | recent(rolling)].
        # Positions: sink->0..sink-1, the R recent frames-> tpos-R..tpos-1, current-> tpos, with
        # tpos=min(frame_index,cap). For k<=kv_window (no eviction) the positions are consecutive
        # 0..k == the absolute path (re-RoPE at the same pos = identical) -> exact match there. ──
        sink, Kr, cap = kv_cache["sink_size"], kv_cache["kv_window"], kv_cache["rope_cap"]
        hw = Hl * Wl
        tpos = min(frame_index, cap)
        q = rope(q, Fl, Hl, Wl, f_start=tpos)
        k_hist, v_hist = kv_cache["k"][layer_idx], kv_cache["v"][layer_idx]
        parts_k, parts_v = [], []
        if k_hist is not None:
            n_cached = k_hist.shape[1] // hw
            n_sink = min(sink, n_cached)          # early frames: fewer than `sink` committed yet
            R = n_cached - n_sink
            if n_sink > 0:
                parts_k.append(rope(k_hist[:, :n_sink * hw], n_sink, Hl, Wl, f_start=0))
                parts_v.append(v_hist[:, :n_sink * hw])
            if R > 0:
                parts_k.append(rope(k_hist[:, n_sink * hw:], R, Hl, Wl, f_start=tpos - R))
                parts_v.append(v_hist[:, n_sink * hw:])
        parts_k.append(rope(k, Fl, Hl, Wl, f_start=tpos))    # current K at tpos
        parts_v.append(v)
        o = sdpa(q, torch.cat(parts_k, 1), torch.cat(parts_v, 1), attn_mask=None).reshape(B, S, n * d)
        if commit:
            nk = k if k_hist is None else torch.cat([k_hist, k], dim=1)   # cache RAW (pre-RoPE)
            nv = v if v_hist is None else torch.cat([v_hist, v], dim=1)
            if nk.shape[1] // hw > sink + Kr:                # evict the OLDEST non-sink frame
                nk = torch.cat([nk[:, :sink * hw], nk[:, (sink + 1) * hw:]], dim=1)
                nv = torch.cat([nv[:, :sink * hw], nv[:, (sink + 1) * hw:]], dim=1)
            kv_cache["k"][layer_idx] = nk.detach()
            kv_cache["v"][layer_idx] = nv.detach()
        return self.o(o)


class CrossAttention(nn.Module):
    """Per-frame per-entity text cross-attention (handoff-doc C9).

    If `context` is [B, Fl, L_text, dim] (per-frame DIFFERENT prompts) we fold the
    frame dim into batch so each frame's hw tokens attend ONLY to that frame's text.
    If `context` is [B, L_text, dim] (shared) we broadcast it across all frames
    (ablation / Incantation-style single-prompt window).
    """

    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # full-dim QK-norm before head reshape (see SelfAttention note).
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)

    def forward(self, x, context, Fl, hw):
        B, L, dim = x.shape
        n, d = self.num_heads, self.head_dim
        assert L == Fl * hw, f"L={L} must equal Fl*hw={Fl*hw}"

        if context.dim() == 4:
            # per-frame: [B, Fl, L_text, dim] -> [B*Fl, L_text, dim]
            Bc, Fc, Lt, dc = context.shape
            assert Fc == Fl and Bc == B and dc == dim
            xf = x.view(B * Fl, hw, dim)                      # frame-major grouping
            cf = context.reshape(B * Fl, Lt, dim)
            q = self.norm_q(self.q(xf)).view(B * Fl, hw, n, d)
            k = self.norm_k(self.k(cf)).view(B * Fl, Lt, n, d)
            v = self.v(cf).view(B * Fl, Lt, n, d)
            o = sdpa(q, k, v).reshape(B * Fl, hw, n * d)
            o = o.view(B, L, dim)
        else:
            # shared: [B, L_text, dim]
            Lt = context.shape[1]
            q = self.norm_q(self.q(x)).view(B, L, n, d)
            k = self.norm_k(self.k(context)).view(B, Lt, n, d)
            v = self.v(context).view(B, Lt, n, d)
            o = sdpa(q, k, v).reshape(B, L, n * d)
        return self.o(o)


# ───────────────────────── block / head ─────────────────────────

class DiTBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = layernorm(dim, affine=False, eps=eps)
        self.norm2 = layernorm(dim, affine=False, eps=eps)
        self.norm3 = layernorm(dim, affine=True, eps=eps)   # cross_attn_norm=True
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, t_mod, context, rope, Fl, Hl, Wl, hw, attn_mask=None,
                kv_cache=None, layer_idx=None, frame_index=None, commit=False,
                tf=False, Fl_cross=None, xcache_probe=None):
        # t_mod is the adaLN modulation, in one of two ranks (WEIGHT-IDENTICAL):
        #   [B, 6, dim]         single timestep per clip  -> broadcast over the sequence
        #   [B, F*hw, 6, dim]   per-frame/per-token (TI2V) -> applied element-wise per token
        # The per-token form lets TI2V mark clean conditioning frames (t=0) vs noisy
        # target frames (t) within one clip. Matches DiffSynth WanModel DiTBlock has_seq.
        # tf=True => the sequence is the TEACHER-FORCING 2·F stream [clean F | noisy F];
        # `Fl` is the TRUE per-clip frame count (used for per-half RoPE), `Fl_cross` is the
        # effective frame count for the per-frame cross-attn (= 2·Fl in TF mode).
        if Fl_cross is None:
            Fl_cross = Fl
        if t_mod.ndim == 4:
            e = [c.squeeze(2) for c in (self.modulation.unsqueeze(1) + t_mod).chunk(6, dim=2)]
        else:
            e = (self.modulation + t_mod).chunk(6, dim=1)   # 6 x [B,1,dim]
        # self-attn (attn_mask = block-causal mask for full-clip causal student; None=bidir
        # OR per-frame KV-cache via kv_cache/layer_idx/frame_index/commit)
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], rope, Fl, Hl, Wl, attn_mask,
                           kv_cache=kv_cache, layer_idx=layer_idx,
                           frame_index=frame_index, commit=commit, tf=tf)
        x = x + y * e[2]                                    # ← x' : post-self-attn hidden state (X-Cache(a) split point)
        # ── X-Cache (a) hook: self-attn has ALREADY run in full (reading the current real KV); x' in hand.
        #    Entered ONLY when a probe is passed; when xcache_probe is None (default / training / xc_on=False)
        #    neither `if` below is entered and 331-340 stay byte-for-byte identical to the original block. ──
        if xcache_probe is not None:
            _res = xcache_probe(x)                           # single-arg = QUERY: hit→cross+FFN residual tensor; miss→None
            if _res is not None:
                return x + _res                             # HIT: x' + cached (cross+FFN) residual, skips 336/338/339
            _x_prime = x                                    # MISS: keep x' so we can register the residual after computing
        # per-frame cross-attn (Fl_cross = 2·Fl when TF so clean+noisy halves each map to their text)
        x = x + self.cross_attn(self.norm3(x), context, Fl_cross, hw)
        # ffn
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        if xcache_probe is not None:
            xcache_probe(_x_prime, x)                        # two-arg = REGISTER: res=(x_out-x').detach() + fingerprint φ(x')
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = layernorm(dim, affine=False, eps=eps)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t):
        # t is the RAW time embedding (not the 6-chunk t_mod), in one of two ranks:
        #   [B, dim]         single per clip  -> broadcast over the sequence
        #   [B, F*hw, dim]   per-frame/per-token (TI2V) -> per-token
        if t.ndim == 3:
            e = [c.squeeze(2) for c in
                 (self.modulation.unsqueeze(0) + t.unsqueeze(2)).chunk(2, dim=2)]
        else:
            e = (self.modulation + t.unsqueeze(1)).chunk(2, dim=1)   # 2 x [B,1,dim]
        return self.head(self.norm(x) * (1 + e[1]) + e[0])


def sinusoidal_embedding_1d(dim, position):
    half = dim // 2
    position = position.float()
    div = torch.exp(-math.log(10000) * torch.arange(half, device=position.device).float() / half)
    ang = position[:, None] * div[None, :]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)   # [P, dim]


# ───────────────────────── model ─────────────────────────

class GameMasterDiT(nn.Module):
    """Bidirectional Wan2.2 DiT + per-frame per-entity text cross-attention."""

    def __init__(
        self,
        dim=3072, in_dim=48, out_dim=48, ffn_dim=14336, text_dim=4096,
        freq_dim=256, num_heads=24, num_layers=30, patch_size=(1, 2, 2),
        eps=1e-6, max_rope_pos=1024, zero_init_head=True, causal=False,
        train_local_attn_size=0, train_sink_size=0,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)
        self.num_heads = num_heads
        self.zero_init_head = zero_init_head
        # causal=False -> the bidirectional teacher (unchanged). causal=True -> the
        # streaming STUDENT: block-causal self-attn (frame k attends frames 0..k,
        # spatial bidir within a frame), SAME RoPE/C9-text/weights. The only diff
        # teacher<->student is this mask; ODE-init reconciles bidir->causal.
        self.causal = causal
        # ── WINDOWED-TRAINING toggle (CF long_video _prepare_blockwise_causal_attn_mask
        # :602-606): a TRAILING window of `train_local_attn_size` frames + an optional
        # `train_sink_size` of always-visible early frames, applied to the PARALLEL training
        # mask only. RoPE stays fully ABSOLUTE (we only MASK far keys; CF avoids the
        # capped-RoPE position-collapse by never capping RoPE in the parallel forward).
        # train_local_attn_size<=0 ⇒ OFF ⇒ byte-identical full-causal mask.
        self.train_local_attn_size = int(train_local_attn_size)
        self.train_sink_size = int(train_sink_size)
        # per-frame cross-attn maps one prompt to one latent TOKEN-frame; with a
        # temporal patch > 1 the number of token-frames != input frames, so the
        # per-frame context length would no longer match. TI2V-5B uses pt=1.
        assert self.patch_size[0] == 1, (
            "per-frame cross-attn requires temporal patch pt==1 "
            f"(got patch_size={self.patch_size})")

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, ffn_dim, num_heads, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        self.rope = Rope3D(dim // num_heads, max_pos=max_rope_pos)
        self.use_gradient_checkpointing = False   # set True for 5B multi-GPU training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # zero-init output head (Wan convention) gives identity-like start when
        # FINETUNING from a pretrained checkpoint. For from-scratch training it
        # zeroes the gradient to every pre-head param on step 0, so disable it.
        if self.zero_init_head:
            nn.init.zeros_(self.head.head.weight)
            nn.init.zeros_(self.head.head.bias)

    def forward(self, x, timestep, context, frame_index=None, kv_cache=None, commit=True,
                clean_x=None, aug_t=None, step_index=None, xcache=None):
        """
        x:        [B, in_dim, Fl, Hl, Wl]  noisy latent (Hl,Wl already VAE-downsampled)
        timestep: [B]  single timestep per clip (t2v SFT)  OR
                  [B, Fl]  per-frame timestep (TI2V: clean cond frames -> t=0)
        context:  [B, Fl, L_text, text_dim]  per-frame T5 embeddings  (per-frame mode)
                  OR [B, L_text, text_dim]    shared embedding         (ablation mode)
        clean_x:  [B, in_dim, Fl, Hl, Wl]  CLEAN GT history channel for TEACHER FORCING.
                  None (default) ⇒ every line below is skipped ⇒ BYTE-IDENTICAL to today.
                  When given, CF-style: a clean prefix is patch-embedded and concatenated
                  BEFORE the noisy stream, the noisy frame attends only the CLEAN past +
                  itself (TF mask), the head/loss read ONLY the noisy half.
        aug_t:    [B] or [B, Fl]  the clean half's timestep (defaults to 0 == exact-clean,
                  CF aug_t=None). >0 ⇒ small history-noise augmentation level.
        returns:  [B, out_dim, Fl, Hl, Wl]  predicted velocity

        When kv_cache is given, routes to the single-frame KV-cache streaming path
        (forward_frame) THROUGH forward() so FSDP's param all-gather fires correctly.
        """
        if kv_cache is not None:
            assert clean_x is None, "teacher-forcing (clean_x) is a train-only path; not for KV-cache rollout"
            return self.forward_frame(x, timestep, context, frame_index, kv_cache, commit,
                                      step_index=step_index, xcache=xcache)
        B = x.shape[0]

        # patchify -> frame-major token sequence [B, L=Fl*hw, dim]
        h = self.patch_embedding(x)                      # [B, dim, Fl', Hl', Wl']
        Fl, Hl, Wl = h.shape[2], h.shape[3], h.shape[4]
        hw = Hl * Wl
        h = h.flatten(2).transpose(1, 2)                 # [B, Fl*hw, dim]

        # ── TEACHER-FORCING: prepend a CLEAN-history token channel (CF causal_model.py:952-963) ──
        tf = clean_x is not None
        S_half = Fl * hw                                 # length of one (clean or noisy) half
        if tf:
            hc = self.patch_embedding(clean_x).flatten(2).transpose(1, 2)   # [B, Fl*hw, dim] clean half
            assert hc.shape == h.shape, "clean_x must produce the same token shape as x"
            h = torch.cat([hc, h], dim=1)                                   # [B, 2*Fl*hw, dim]  CLEAN | NOISY

        # time embedding. timestep is either:
        #   [B]      single timestep per clip (t2v / standard flow-matching SFT)
        #   [B, Fl]  per-frame timestep (TI2V: clean conditioning frames carry t=0)
        # The per-frame path expands to per-token (frame-major f*hw) and is WEIGHT-
        # IDENTICAL to the [B] path (same time_embedding / time_projection params).
        # In TF mode the stream is per-token-heterogeneous (clean half=aug_t, noisy half=t),
        # so we FORCE the per-frame path with a [B, 2·Fl] timestep (clean ts || noisy ts).
        if tf:
            if timestep.ndim == 1:
                ts_noisy = timestep[:, None].expand(B, Fl)                  # [B,Fl]
            else:
                ts_noisy = timestep                                        # already [B,Fl]
            if aug_t is None:
                ts_clean = torch.zeros_like(ts_noisy)                      # exact clean (CF aug_t=None ⇒ 0)
            elif aug_t.ndim == 1:
                ts_clean = aug_t[:, None].expand(B, Fl)
            else:
                ts_clean = aug_t                                           # [B,Fl]
            timestep = torch.cat([ts_clean, ts_noisy], dim=1)             # [B, 2*Fl]
        F_tok = 2 * Fl if tf else Fl                                       # token frames in the stream
        if timestep.ndim == 1:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(h.dtype))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))   # [B,6,dim]
        else:
            Bt, Ft = timestep.shape
            assert Ft == F_tok, f"per-frame timestep F={Ft} != stream frames {F_tok}"
            ts_tok = timestep.unsqueeze(-1).expand(Bt, F_tok, hw).reshape(Bt * F_tok * hw)
            tt = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, ts_tok).to(h.dtype))
            t = tt.view(Bt, F_tok * hw, self.dim)                         # [B, F_tok*hw, dim]
            t_mod = self.time_projection(t).unflatten(2, (6, self.dim))   # [B, F_tok*hw, 6, dim]

        # text embedding (works for both [B,Lt,4096] and [B,Fl,Lt,4096])
        ctx = self.text_embedding(context)
        # TF: the clean half needs a text channel too — reuse the SAME per-frame prompts
        # (clean | noisy share text), matching CF passing the same conditional_dict.
        if tf and ctx.dim() == 4:
            ctx = torch.cat([ctx, ctx], dim=1)                           # [B, 2*Fl, Lt, dim]

        # block-causal self-attn mask for the streaming student (None for the teacher).
        # frame-major tokens: token i (frame fi=i//hw) may attend token j iff fj<=fi.
        attn_mask = None
        if self.causal:
            if tf:
                # ── TEACHER-FORCING mask over the 2·F stream (CF _prepare_teacher_forcing_mask
                # :627-637, specialized to num_frame_per_block=1). tokens [0,S)=clean, [S,2S)=noisy.
                # clean q: block-causal WITHIN the clean half (sees clean 0..f). noisy q frame f:
                # sees clean 0..f-1 (strictly-prev) + itself; never other noisy, never clean f. ──
                S = Fl * hw
                tok = torch.arange(2 * S, device=h.device)
                cln = tok < S                                            # clean-half membership
                fr = torch.where(cln, tok // hw, (tok - S) // hw)        # frame idx 0..Fl-1 for both halves
                if self.train_local_attn_size and self.train_local_attn_size > 0:
                    W = self.train_local_attn_size
                else:
                    W = None
                cq, ck = cln[:, None], cln[None, :]
                fq, fk = fr[:, None], fr[None, :]
                clean_mask = cq & ck & (fk <= fq)                        # clean q: block-causal in clean half
                noise_self = (~cq) & (~ck) & (fq == fk)                  # C1: own noisy frame
                noise_ctx = (~cq) & ck & (fk < fq)                       # C2: strictly-prev CLEAN frames
                if W is not None:
                    # windowed-training (C9): bound C2's clean-past to a trailing window + sink.
                    in_win = (fk > (fq - W))
                    if self.train_sink_size and self.train_sink_size > 0:
                        in_win = in_win | (fk < self.train_sink_size)
                    noise_ctx = noise_ctx & in_win
                eye = tok[:, None] == tok[None, :]
                attn_mask = eye | clean_mask | noise_self | noise_ctx    # [2S,2S] bool, True=attend
            else:
                fidx = torch.arange(Fl * hw, device=h.device) // hw      # [S] frame index
                causal = fidx.unsqueeze(1) >= fidx.unsqueeze(0)          # [S,S] bool, True=attend
                if self.train_local_attn_size and self.train_local_attn_size > 0:
                    # ── WINDOWED TRAINING (CF :602-606): trailing window of W frames + optional
                    # sink of always-visible early frames; diagonal always kept. RoPE stays absolute. ──
                    W = self.train_local_attn_size
                    in_win = fidx.unsqueeze(0) > (fidx.unsqueeze(1) - W)
                    if self.train_sink_size and self.train_sink_size > 0:
                        in_win = in_win | (fidx.unsqueeze(0) < self.train_sink_size)
                    attn_mask = (causal & in_win) | (fidx.unsqueeze(1) == fidx.unsqueeze(0))
                else:
                    attn_mask = causal

        Fl_cross = F_tok                                                 # cross-attn frame count (2·Fl in TF)
        for blk in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                h = torch_checkpoint(
                    blk, h, t_mod, ctx, self.rope, Fl, Hl, Wl, hw, attn_mask,
                    None, None, None, False, tf, Fl_cross,
                    use_reentrant=False)
            else:
                h = blk(h, t_mod, ctx, self.rope, Fl, Hl, Wl, hw, attn_mask,
                        tf=tf, Fl_cross=Fl_cross)

        if tf:
            # drop the clean prefix; head/loss on the noisy half only (CF :997-998).
            h = h[:, S_half:]
            t = t[:, S_half:] if t.ndim == 3 else t                      # head modulation: noisy-half time only
        h = self.head(h, t)                              # [B, L, out_dim*prod(patch)]
        return self._unpatchify(h, Fl, Hl, Wl)

    # ─────────── causal-student KV-cache streaming forward (for DMD rollout + inference) ───────────

    def init_kv_cache(self, num_layers=None, kv_window=None, rope_cap=16, sink_size=1):
        """Fresh per-rollout cache (one clip). Reset per new clip.

        kv_window=None -> ABSOLUTE mode (training/short eval; cache post-RoPE K at absolute
        frame_index, grows unbounded, RoPE OOD past the trained F). Set kv_window (e.g. 7) +
        rope_cap (e.g. 16) + sink_size (1) -> SLIDING mode for stable LONG rollout: cache raw K,
        re-RoPE at bounded local positions, evict oldest non-sink beyond sink+kv_window."""
        n = num_layers if num_layers is not None else len(self.blocks)
        c = {"k": [None] * n, "v": [None] * n, "frames_cached": 0}
        if kv_window is not None:
            c.update(kv_window=kv_window, rope_cap=rope_cap, sink_size=sink_size)
        return c

    def init_xcache(self, num_layers=None, tau_floor=0.97, margin=0.02,
                    alpha=0.3, max_dev_frac=0.20):
        """Fresh per-rollout X-Cache (training-free cross-frame block-residual reuse). Reset per clip,
        exactly like init_kv_cache. Analogous state to the KV cache but keyed by (step_index, block_idx)
        instead of layer_idx, so each denoise step keeps its OWN reference frame.

        Contents:
          residual[(step_index, block_idx)]    -> DETACHED block residual (h_out - h_in) from the last
                                                  frame this block was actually COMPUTED at that step.
          fingerprint[(step_index, block_idx)] -> DETACHED block INPUT h from that same last-compute
                                                  frame (the reference we measure drift against).
          tau[block_idx]                        -> per-block EMA-smoothed cosine HIT threshold.
        Tunables (see risks — need on-GPU sweep): tau_floor (min cos to ever reuse), margin (subtracted
        from the running similarity to set the adaptive threshold), alpha (EMA rate), max_dev_frac
        (secondary guard: relative max abs deviation of the input must stay under this to reuse)."""
        n = num_layers if num_layers is not None else len(self.blocks)
        return {
            "residual": {},                 # (step_index, block_idx) -> residual tensor
            "fingerprint": {},              # (step_index, block_idx) -> input-h reference tensor
            "tau": _XCACHE_TAU,             # FIXED cosine hit threshold (matches probe; GM_XCACHE_TAU)
            "tau_floor": float(tau_floor),
            "margin": float(margin),
            "alpha": float(alpha),
            "max_dev_frac": float(max_dev_frac),
            "hits": 0, "misses": 0,         # cheap counters for A/B reporting
        }

    def forward_frame(self, x_k, timestep_k, context_k, frame_index, kv_cache, commit=True,
                      step_index=None, xcache=None):
        """Generate/score ONE frame attending the cached history (frames 0..frame_index-1)
        plus itself. NUMERICALLY EQUIVALENT to the full-clip causal forward at this frame
        (consecutive absolute RoPE pos = frame_index, post-RoPE K cached, frame sees all 0..k).
          x_k:        [B, in_dim, 1, Hpix, Wpix]  one latent frame
          timestep_k: [B] or [B,1]   this frame's denoise timestep
          context_k:  [B, 1, L_text, text_dim] (per-frame) or [B, L_text, text_dim]
          frame_index: int absolute temporal index k
          commit: True persists this frame's K/V into the cache (clean/final pass); False = a
                  transient denoise step (read cache, don't pollute it with the noisy frame).
          step_index: which denoise step (0..len(denoise_list)-1) this pass is; keys the X-Cache so
                  each step keeps its own reference frame. None (default) ⇒ X-Cache fully off.
          xcache: the dict from init_xcache(), or None (default). When None OR GM_XCACHE unset OR
                  commit=True, EVERY block is computed byte-for-byte (original path). Only on
                  (xcache is not None AND GM_XCACHE AND not commit) may a block be skipped.
        Returns predicted velocity [B, out_dim, 1, Hpix, Wpix]."""
        assert self.causal, "forward_frame is the causal-student streaming path (use causal=True)"
        # RoPE table bounds the ABSOLUTE-mode position (f_start=frame_index). The SLIDING-window path
        # clamps every position to <= rope_cap (tpos=min(frame_index,cap)) + a sink at pos 0, so f_t is
        # only ever indexed in [0, rope_cap]; frame_index may grow UNBOUNDED there -> long rollout past
        # max_rope_pos (deploy uses sliding). Only the absolute-KV path actually indexes f_t[frame_index].
        if kv_cache.get("kv_window") is None:
            assert frame_index < self.rope.f_t.shape[0], \
                f"frame_index {frame_index} >= max_rope_pos {self.rope.f_t.shape[0]} (absolute-KV mode)"
        B = x_k.shape[0]
        h = self.patch_embedding(x_k)                    # [B, dim, 1, Hl, Wl]
        Fl, Hl, Wl = 1, h.shape[3], h.shape[4]
        hw = Hl * Wl
        h = h.flatten(2).transpose(1, 2)                 # [B, hw, dim]
        # single timestep for this frame -> [B] time path (weight-identical to full-clip per-frame)
        ts = timestep_k.reshape(B).to(h.dtype) if timestep_k.dtype.is_floating_point \
            else timestep_k.reshape(B).float()
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, ts).to(h.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))   # [B,6,dim]
        ctx = self.text_embedding(context_k)             # [B,1,Lt,dim] or [B,Lt,dim]
        # X-Cache is only eligible on the transient denoise passes (commit=False). Rationale:
        #   • commit=True passes WRITE the KV cache — they must run every block's self-attn in full,
        #     or the cached K/V (and thus all later frames) would be corrupted.
        #   • On commit=False the frame's output is a throwaway velocity estimate and NOTHING is
        #     written to the KV cache, so skipping a block (and, with it, that block's self-attn KV
        #     READ) is safe: the skipped block's contribution is replaced by the cached residual, and
        #     because we never commit here the cache is left byte-identical to the no-skip path.
        # v3 (2026-07-22) stopgap fix for the artifact collapse: only the HIGH-noise denoise pass
        # (step_index==0, t=1000) may skip. The LOW-noise pass (step_index==1, t=250) PRODUCES the
        # frame `den` that is then committed into the rolling KV — it must be fully computed so a
        # dirty (skip-approximated) frame never pollutes the KV history (unbounded error accumulation,
        # = the paper's Table-3 collapse when the KV-update pass isn't force-computed). commit=True
        # (the pure KV writer) is already forced full below.
        xc_on = (xcache is not None) and _XCACHE and (not commit) and (step_index == 0)
        for li, blk in enumerate(self.blocks):
            if not xc_on:
                # ── UNCHANGED original path (xcache None / GM_XCACHE off / commit=True) ──
                h = blk(h, t_mod, ctx, self.rope, Fl, Hl, Wl, hw, attn_mask=None,
                        kv_cache=kv_cache, layer_idx=li, frame_index=frame_index, commit=commit)
                continue
            # ── X-Cache (a) (2026-07-22): self-attn ALWAYS runs in full (reads the current real sliding
            #    KV), only the cross-attn+FFN residual is cached, and the fingerprint is built on x'
            #    (post-self-attn) so the gate sees the state with THIS frame's KV already applied. The
            #    block computes x' internally and calls _probe back: single-arg = query (hit→return
            #    cross+FFN residual, block returns x'+res skipping cross/FFN; miss→None), two-arg =
            #    register the miss (res=(x_out-x').detach(), fp=φ(x')). hits/misses/tau/(step_index,li)
            #    key / _xcache_fp are all unchanged so the A/B report scripts need no edits. ──
            key = (step_index, li)
            _fp_box = {}                                    # closure scratch: query stashes φ(x'), register reads it back
            def _probe(*args, _key=key, _box=_fp_box):
                if len(args) == 1:
                    # QUERY: arg is x' (post-self-attn). Fingerprint now on x'; gate otherwise as v2/probe.
                    xp = args[0]
                    fp = _xcache_fp(xp, Hl, Wl)             # φ(x') — reuse existing fingerprint, input swapped to x'
                    _box["fp"] = fp
                    prev_fp = xcache["fingerprint"].get(_key)
                    prev_res = xcache["residual"].get(_key)
                    if prev_fp is not None and prev_res is not None and prev_fp.shape == fp.shape:
                        cos = float(F.cosine_similarity(fp, prev_fp, dim=0).clamp(-1, 1).item())
                        if cos >= xcache["tau"]:
                            xcache["hits"] += 1
                            return prev_res                 # HIT: residual holding only cross+FFN
                    return None                             # MISS / no reference: let the block compute cross+FFN
                # REGISTER (two-arg): args=(x', x_out). Residual holds only cross+FFN; reference fp=φ(x').
                xp, xout = args
                xcache["residual"][_key] = (xout - xp).detach()
                xcache["fingerprint"][_key] = _box["fp"]
                xcache["misses"] += 1
                return None
            h = blk(h, t_mod, ctx, self.rope, Fl, Hl, Wl, hw, attn_mask=None,
                    kv_cache=kv_cache, layer_idx=li, frame_index=frame_index, commit=commit,
                    xcache_probe=_probe)
        h = self.head(h, t)
        if commit:
            kv_cache["frames_cached"] = frame_index + 1
        return self._unpatchify(h, Fl, Hl, Wl)

    def _unpatchify(self, x, Fl, Hl, Wl):
        pt, ph, pw = self.patch_size
        c = self.out_dim
        B = x.shape[0]
        x = x.view(B, Fl, Hl, Wl, pt, ph, pw, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(B, c, Fl * pt, Hl * ph, Wl * pw)


# Convenience tiny config for fast CPU/GPU smoke tests.
def tiny_config():
    return dict(
        dim=128, in_dim=48, out_dim=48, ffn_dim=256, text_dim=4096,
        freq_dim=256, num_heads=4, num_layers=2, patch_size=(1, 2, 2),
    )


def small_config():
    # bigger than tiny so an overfit smoke run has enough capacity to drive the
    # flow-matching loss well down (still tiny vs the 5B real model; runs in secs).
    return dict(
        dim=256, in_dim=48, out_dim=48, ffn_dim=768, text_dim=4096,
        freq_dim=256, num_heads=8, num_layers=4, patch_size=(1, 2, 2),
    )


def ti2v_5b_config():
    return dict(
        dim=3072, in_dim=48, out_dim=48, ffn_dim=14336, text_dim=4096,
        freq_dim=256, num_heads=24, num_layers=30, patch_size=(1, 2, 2),
    )
