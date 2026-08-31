"""Chained bidirectional rollout: window N's last frame seeds window N+1.

Stress-tests what the user cares about at v1 completion:
  - style persistence over ~20 s (4 chained 101-frame windows, far beyond
    anything the model saw in training);
  - collision behavior (pick a first frame facing a wall/fence and hold W).

Usage:
  python scripts/eval_chain.py --ckpt step-30000.safetensors \
      --image scene.png --name village --out_dir eval_chain30k \
      [--windows 4] [--keys W] [--action_cfg 1.0]
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image")
    p.add_argument("--name")
    p.add_argument("--jobs", default=None,
                   help="JSON file: [{name, image, full_actions|keys}, ...] — "
                        "runs every job with a single pipeline load")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--keys", default="W",
                   help="comma list held for the whole run, OR per-window specs "
                        "separated by '/', e.g. 'W/W/S/S' for an out-and-back")
    p.add_argument("--full_actions", default=None,
                   help="parquet with >= windows*100+1 rows of per-frame actions; "
                        "window w consumes rows [100w, 100w+101). Overrides --keys.")
    p.add_argument("--base_model", default="/nfs/zeqingwang/models/base_model")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--action_cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--action_context_table", default=None,
                   help="Enable per-frame action-context (table) mode.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.jobs:
        import copy
        import json as _json
        jobs = _json.load(open(args.jobs))
        pipe_holder = {}
        for job in jobs:
            sub = copy.copy(args)
            sub.jobs = None
            sub.name = job["name"]
            sub.image = job["image"]
            sub.full_actions = job.get("full_actions")
            if job.get("keys"):
                sub.keys = job["keys"]
            run_one(sub, pipe_holder)
        return
    run_one(args, {})


def run_one(args, pipe_holder):
    parqs = []
    if args.full_actions:
        full = pd.read_parquet(args.full_actions)
        assert len(full) >= args.windows * 100 + 1, "full_actions too short"
        for w in range(args.windows):
            seg = full.iloc[w * 100: w * 100 + NF].reset_index(drop=True)
            parq = os.path.join(args.out_dir, f"{args.name}_w{w}_gt.parquet")
            seg.to_parquet(parq)
            parqs.append(parq)
    else:
        specs = args.keys.split("/")
        per_window = [specs[min(w, len(specs) - 1)] for w in range(args.windows)]
        for w, spec in enumerate(per_window):
            act = np.zeros((NF, len(COLS)), np.float32)
            for k in spec.split(","):
                act[:, COLS.index(k.strip())] = 1
            parq = os.path.join(args.out_dir, f"{args.name}_w{w}_{spec.replace(',', '_')}.parquet")
            pd.DataFrame(act, columns=COLS).to_parquet(parq)
            parqs.append(parq)

    pipe = SFPipeline.from_pretrained(
        base_model_dir=args.base_model, checkpoint_path=args.ckpt,
        variant="vrising", torch_dtype=torch.bfloat16,
    ).to("cuda")
    if getattr(args, "action_context_table", None):
        pipe.load_action_context_table(args.action_context_table)

    frames_all = []
    img = Image.open(args.image)
    for w in range(args.windows):
        out = pipe(image=img, actions_parquet=parqs[w],
                   num_frames=NF, num_inference_steps=args.steps,
                   cfg_scale=args.cfg, action_cfg_scale=args.action_cfg,
                   seed=args.seed + w)
        frames = out.frames[0]
        # window w reuses the previous window's last frame as frame 0 — drop
        # the duplicate when concatenating
        frames_all.extend(frames if w == 0 else frames[1:])
        img = frames[-1]
        print(f"[chain] {args.name}: window {w + 1}/{args.windows} done "
              f"({len(frames_all)} frames)", flush=True)

    out_path = os.path.join(args.out_dir, f"{args.name}_chain{args.windows}.mp4")
    save_mp4(frames_all, out_path, fps=20)
    print(f"[chain] saved {out_path} ({len(frames_all) / 20:.1f}s)", flush=True)

    # burn the action HUD (concatenated per-window, dropping dup boundary rows)
    from vrising_data.verify import overlay_clip
    tabs = [pd.read_parquet(q) for q in parqs]
    cat = pd.concat([tabs[0]] + [t.iloc[1:] for t in tabs[1:]], ignore_index=True)
    catp = os.path.join(args.out_dir, f"{args.name}_actions_cat.parquet")
    cat.to_parquet(catp)
    hud = out_path.replace(".mp4", "_hud.mp4")
    overlay_clip(out_path, catp, hud)
    print(f"[chain] saved {hud}", flush=True)


if __name__ == "__main__":
    main()
