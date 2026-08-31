"""Action-control eval: one checkpoint, N first-frames x M action scripts.

Loads the pipeline once, generates every combination with a fixed seed, and
writes <out_dir>/<img>__<action>.mp4. The point is visual: identical first
frame + different WASD scripts should yield clearly divergent trajectories.

Usage:
  python scripts/eval_grid.py --ckpt runs/.../step-3000.safetensors \
      --images f1.png f2.png --out_dir eval_step3000 [--action_cfg 1.5]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ReactiveGWM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ReactiveGWM_Code.inference import SFPipeline
from ReactiveGWM_Code.inference.utils import save_mp4

COLS = ["W", "A", "S", "D", "MOUSE0", "CAM_X", "CAM_Y", "CAM_ACTIVE"]
NF = 101


def scripts():
    def blank():
        return np.zeros((NF, len(COLS)), np.float32)

    out = {}
    for k in ("W", "A", "S", "D"):
        a = blank(); a[:, COLS.index(k)] = 1
        out[f"hold_{k}"] = a
    a = blank(); a[:50, COLS.index("W")] = 1; a[50:, COLS.index("D")] = 1
    out["W_then_D"] = a
    out["idle"] = blank()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_model", default="/nfs/zeqingwang/models/base_model")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--action_cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    act_dir = os.path.join(args.out_dir, "actions")
    os.makedirs(act_dir, exist_ok=True)
    parqs = {}
    for name, arr in scripts().items():
        path = os.path.join(act_dir, f"{name}.parquet")
        pd.DataFrame(arr, columns=COLS).to_parquet(path)
        parqs[name] = path

    pipe = SFPipeline.from_pretrained(
        base_model_dir=args.base_model,
        checkpoint_path=args.ckpt,
        variant="vrising",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    for img_path in args.images:
        stem = Path(img_path).stem
        img = Image.open(img_path)
        for name, parq in parqs.items():
            out_path = os.path.join(args.out_dir, f"{stem}__{name}.mp4")
            if os.path.exists(out_path):
                print(f"[skip] {out_path}", flush=True)
                continue
            out = pipe(
                image=img, actions_parquet=parq,
                num_frames=NF, num_inference_steps=args.steps,
                cfg_scale=args.cfg, action_cfg_scale=args.action_cfg,
                seed=args.seed,
            )
            save_mp4(out.frames[0], out_path, fps=20)
            print(f"[eval] {out_path}", flush=True)


if __name__ == "__main__":
    main()
