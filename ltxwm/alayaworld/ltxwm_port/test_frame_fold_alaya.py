"""Bit-exactness + frame-isolation tests for the alayaworld frame-fold port.

Reference = python loop calling the ORIGINAL _apply_text_cross_attention once
per frame on that frame's token/context slice. Folded output must match the
loop bitwise-close (fp32 CPU), and per-frame context edits must only affect
that frame's tokens. Global (non F*L) contexts must pass through unchanged.
"""
import sys

import torch

sys.path.insert(0, "/data/yuzhewu/ltxwm/alayaworld")
sys.path.insert(0, "/data/yuzhewu/ltxwm/alayaworld/ltxwm_port")

from ltx2.modules.attention import AttentionFunction
from ltx2.modules.model_ltx_2_3 import LTX23AttentionBlock, TransformerConfig
from frame_context_patch_alaya import install_frame_context, _ORIG

torch.manual_seed(0)

B, F, TPF, L = 2, 4, 6, 5
DIM, HEADS, DHEAD, CTX_DIM = 64, 4, 16, 64
T = F * TPF


def make_block(adaln: bool) -> LTX23AttentionBlock:
    cfg = TransformerConfig(dim=DIM, heads=HEADS, d_head=DHEAD,
                            context_dim=CTX_DIM, cross_attention_adaln=adaln)
    blk = LTX23AttentionBlock(idx=0, video=cfg, attention_function=AttentionFunction.PYTORCH)
    with torch.no_grad():
        for p in blk.parameters():
            torch.nn.init.normal_(p, std=0.02)
    return blk.float().eval()


def run_case(adaln: bool):
    blk = make_block(adaln)
    sst = 9 if adaln else 6
    x = torch.randn(B, T, DIM)
    ctx = torch.randn(B, F * L, CTX_DIM)
    ts = torch.randn(B, T, sst * DIM)
    pts = torch.randn(B, 1, 2 * DIM) if adaln else None

    # reference: per-frame loop through the ORIGINAL method
    ref = []
    for k in range(F):
        ref.append(_ORIG(blk, x[:, k * TPF:(k + 1) * TPF], ctx[:, k * L:(k + 1) * L],
                         ts[:, k * TPF:(k + 1) * TPF], pts, None))
    ref = torch.cat(ref, dim=1)

    n = install_frame_context(blk, F, L)
    assert n == 1, n
    out = blk._apply_text_cross_attention(x, ctx, ts, pts, None)
    diff = (out - ref).abs().max().item()
    print(f"adaln={adaln}: fold-vs-loop max diff {diff:.2e}")
    assert diff < 1e-5, diff

    # frame isolation: perturb frame 2's context -> only frame 2's tokens change
    ctx2 = ctx.clone()
    ctx2[:, 2 * L:3 * L] += 1.0
    out2 = blk._apply_text_cross_attention(x, ctx2, ts, pts, None)
    delta = (out2 - out).abs().amax(dim=(0, 2))          # [T]
    changed = delta > 1e-7
    expect = torch.zeros(T, dtype=torch.bool)
    expect[2 * TPF:3 * TPF] = True
    assert torch.equal(changed, expect), (changed.nonzero().flatten(), expect.nonzero().flatten())
    print(f"adaln={adaln}: frame isolation OK")

    # pass-through: global prompt (length != F*L) must equal the original path
    gctx = torch.randn(B, 33, CTX_DIM)
    out_g = blk._apply_text_cross_attention(x, gctx, ts, pts, None)
    ref_g = _ORIG(blk, x, gctx, ts, pts, None)
    assert torch.equal(out_g, ref_g)
    print(f"adaln={adaln}: global-context pass-through OK")


run_case(False)
run_case(True)
print("ALL_FOLD_TESTS_PASS")
