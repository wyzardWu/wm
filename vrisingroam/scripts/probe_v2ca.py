"""W/D probe for the CA bidirectional teacher (table mode).

Runs the calibrated probes through SFPipeline in per-frame action-context
mode and prints phase-correlation displacement.
  W-probe GT (measure_motion): dy=+7.87   D-probe GT: dx=-12.04
Usage:
  python scripts/probe_v2ca.py --ckpt runs/v2_ca_bidir/step-3000.safetensors \
      --table /data/yuzhewu/vrisingroam/processed/action_context_table_v2.pt \
      --out_dir /data/yuzhewu/vrisingroam/eval_v2ca [--action_cfg 1.0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

BASE = "/nfs/zeqingwang/models/base_model"
ET = "/data/yuzhewu/vrisingroam/eval_teacher"
PROBES = [
    ("probeW", f"{ET}/probeW_row5.png",
     "data/processed/combined_5d/actions/chunk_005/20260731_213546_491_chunk_005_00541535.parquet",
     "GT dy=+7.87"),
    ("probeD", f"{ET}/probeD_row64.png",
     "data/processed/combined_5d/actions/chunk_005/20260731_213546_491_chunk_005_00559412.parquet",
     "GT dx=-12.04"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--action_cfg", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    import torch
    from ReactiveGWM_Code.inference import SFPipeline
    from scripts.measure_motion import measure

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(args.ckpt).replace(".safetensors", "")

    pipe = SFPipeline.from_pretrained(BASE, args.ckpt, variant="vrising").to("cuda")
    pipe.load_action_context_table(args.table)

    for name, img, pq, gt in PROBES:
        out = pipe(Image.open(img), pq, num_frames=101,
                   num_inference_steps=args.steps,
                   cfg_scale=1.0, action_cfg_scale=args.action_cfg, seed=0)
        mp4 = os.path.join(args.out_dir, f"{tag}_{name}_acfg{args.action_cfg}.mp4")
        import imageio
        with imageio.get_writer(mp4, fps=20, quality=8) as w:
            for f in out.frames[0]:
                w.append_data(__import__("numpy").asarray(f))
        mag, dx, dy, n = measure(mp4)
        print(f"PROBE {name} ({gt}): |v|={mag:.2f} dx={dx:+.2f} dy={dy:+.2f} "
              f"n={n} -> {mp4}", flush=True)


if __name__ == "__main__":
    main()
