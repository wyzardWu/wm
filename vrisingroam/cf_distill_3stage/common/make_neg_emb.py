"""Embed the negative / empty prompt for DMD classifier-free guidance.

text_table.pt contains ONLY the 135 templated render-view prompts (full + boss-dropped),
NOT the empty string '' that DMD's CFG uses as the unconditional prompt. This script embeds
'' (or any --prompt) with the SAME UMT5-XXL encoder + zero-pad scheme as
scripts/precompute_text.py, so the negative embedding is in-distribution for the teacher.

Saves: distill/neg_emb.pt = {"neg_emb": [512, 4096] bf16, "prompt": "<prompt>"}

Run once on a free GPU:
    CUDA_VISIBLE_DEVICES=<free> \
      /opt/dlami/nvme/miniforge3/envs/gamemaster/bin/python distill/make_neg_emb.py
"""
import argparse
import os
import sys

import torch

GM = os.environ.get("GM") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GM)
sys.path.insert(0, os.environ.get("GM_WAN_DIR", "/opt/dlami/nvme/zhiyangdeng/_shared/vendor/vendor_wan"))   # read-only: modules.t5 (same as precompute_text.py)
CKPT = os.environ.get("GM_WAN_CKPT", "/opt/dlami/nvme/zhiyangdeng/_shared/base_models/wan2.2-ti2v-5b-dit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="", help="negative prompt (config.negative_prompt; default empty)")
    ap.add_argument("--out", default=os.path.join(GM, "distill/neg_emb.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--t5_pth", default=os.path.join(CKPT, "models_t5_umt5-xxl-enc-bf16.pth"))
    ap.add_argument("--tok", default=os.path.join(CKPT, "google/umt5-xxl"))
    args = ap.parse_args()

    from modules.t5 import T5EncoderModel
    t5 = T5EncoderModel(text_len=args.text_len, dtype=torch.bfloat16, device=args.device,
                        checkpoint_path=args.t5_pth, tokenizer_path=args.tok)
    with torch.no_grad():
        outs = t5([args.prompt], args.device)            # list of [Lvar, 4096]
    e = outs[0][:args.text_len]
    t = torch.zeros(args.text_len, 4096, dtype=torch.bfloat16)   # zero-pad exactly like precompute_text.py
    t[: e.shape[0]] = e.to(torch.bfloat16).cpu()
    torch.save({"neg_emb": t, "prompt": args.prompt}, args.out)
    print(f"saved neg_emb {tuple(t.shape)} (prompt={args.prompt!r}) -> {args.out}")


if __name__ == "__main__":
    main()
