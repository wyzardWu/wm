"""Build a merged SF3 ckpt by injecting SF2's trained cross_attn weights.

The SF3 freeze_xattn run never wrote cross_attn keys to its safetensors —
inference therefore falls back to base Wan2.2 cross_attn at load time. This
script overlays the 300 cross_attn keys from a chosen SF2 ckpt onto the SF3
state dict and saves a new safetensors. text_embedding is left as SF3's
(SF2 vs SF3 text_embedding diverge slightly; preserving SF3's keeps the rest
of the SF3 graph self-consistent).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sf3_ckpt", required=True)
    p.add_argument("--sf2_ckpt", required=True)
    p.add_argument("--out_ckpt", required=True)
    args = p.parse_args()

    sf3 = load_file(args.sf3_ckpt)
    sf2 = load_file(args.sf2_ckpt)
    print(f"[load] sf3={len(sf3)} keys, sf2={len(sf2)} keys")

    xa_keys = sorted(k for k in sf2 if "cross_attn" in k)
    print(f"[swap] {len(xa_keys)} cross_attn keys to inject from SF2")

    merged = dict(sf3)
    for k in xa_keys:
        if k in merged:
            raise SystemExit(f"unexpected: SF3 already has key {k}")
        merged[k] = sf2[k].clone()

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, args.out_ckpt)
    print(f"[save] {args.out_ckpt} ({len(merged)} keys)")


if __name__ == "__main__":
    main()
