"""EYBX scene-switch probes for the overfit bidirectional model.

Two probe families (bidirectional sampling, table mode = action ⊕ scene):
  const:  first frame of scene X + constant action + constant scene-X token
          -> checks scene-token/style binding + action following
  switch: first frame of scene A + constant action + scene token flips A->B
          at frame 41 (latent 11) -> THE verdict probe: does the frame-level
          token flip produce a hard cut into scene B with motion continuity?

Synthetic actions parquet: W/A/S/D/MOUSE0/SCENE columns (eybx variant).
Usage:
  python scripts/probe_eybx.py --ckpt runs/eybx_overfit_v3/step-14000.safetensors \
      --out_dir /data/yuzhewu/vrisingroam/eybx_probes [--steps 30] [--action_cfg 1.0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "/nfs/zeqingwang/models/base_model"
TABLES = "/data/yuzhewu/eybxroam/tables"
SEEDS = "/data/yuzhewu/eybxroam/probe_seeds"   # {scene}.png, written by prep step
SCENES = ["infiniteDungeon", "mountainPass", "hills"]
CUT = 41                                       # frame -> latent 11 (trained cut)


def synth_parquet(path, action, scene_a, scene_b=None, cut=CUT, n=101):
    import pandas as pd
    cols = {k: [0.0] * n for k in ["W", "A", "S", "D", "MOUSE0"]}
    for ch in action:
        cols[ch] = [1.0] * n
    sc = [float(scene_a)] * (cut if scene_b is not None else n)
    if scene_b is not None:
        sc += [float(scene_b)] * (n - cut)
    cols["SCENE"] = sc
    pd.DataFrame(cols).to_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="/data/yuzhewu/vrisingroam/eybx_probes")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--action_cfg", type=float, default=1.0)
    ap.add_argument("--jobs_json", default=None,
                    help="optional JSON list of {name, seed(png path), sa, sb(null=const), cut, act}")
    args = ap.parse_args()

    import numpy as np
    import imageio
    import torch  # noqa: F401
    from PIL import Image
    from ReactiveGWM_Code.inference import SFPipeline

    tag = os.path.basename(args.ckpt).replace(".safetensors", "")
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_actions")
    os.makedirs(tmp, exist_ok=True)

    pipe = SFPipeline.from_pretrained(BASE, args.ckpt, variant="eybx").to("cuda")
    pipe.load_action_context_table(f"{TABLES}/eybx_action_table.pt")
    pipe.load_scene_context_table(f"{TABLES}/eybx_scene_table.pt")

    if args.jobs_json:
        import json
        jobs = [(j["name"], j.get("sa"), j.get("sb"), j.get("act", "W"),
                 j["seed"], j.get("cut", CUT), j.get("pq")) for j in json.load(open(args.jobs_json))]
    else:
        jobs = []
        for i, rg in enumerate(SCENES):                   # const probes
            jobs.append((f"const_{rg}", i, None, "W", f"{SEEDS}/{rg}.png", CUT, None))
        jobs += [                                         # switch probes
            ("switch_dungeon2hills", 0, 2, "W", f"{SEEDS}/infiniteDungeon.png", CUT, None),
            ("switch_hills2dungeon", 2, 0, "W", f"{SEEDS}/hills.png", CUT, None),
            ("switch_mp2dungeon", 1, 0, "W", f"{SEEDS}/mountainPass.png", CUT, None),
            ("switch_hills2mp", 2, 1, "W", f"{SEEDS}/hills.png", CUT, None),
        ]
    for name, sa, sb, act, seed_path, cut, ready_pq in jobs:
        if ready_pq:
            pq = ready_pq
        else:
            pq = os.path.join(tmp, f"{name}.parquet")
            synth_parquet(pq, act, sa, sb, cut=cut)
        seed_img = Image.open(seed_path)
        out = pipe(seed_img, pq, num_frames=101,
                   num_inference_steps=args.steps,
                   cfg_scale=1.0, action_cfg_scale=args.action_cfg, seed=0)
        mp4 = os.path.join(out_dir, f"{name}.mp4")
        with imageio.get_writer(mp4, fps=20, quality=8) as w:
            for f in out.frames[0]:
                w.append_data(np.asarray(f))
        print(f"PROBE {name} -> {mp4}", flush=True)


if __name__ == "__main__":
    main()
