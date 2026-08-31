"""GT-replay eval: for each chosen clip, feed its real first frame + its real
action parquet to the checkpoint and render a side-by-side (generated | GT).

Usage:
  python scripts/eval_gt.py --ckpt <step-N.safetensors> --out_dir eval_gtN \
      --clips clips/chunk_020/xxx.mp4 [more...] --data_root <out_root>
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ReactiveGWM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from ReactiveGWM_Code.inference import SFPipeline
from ReactiveGWM_Code.inference.utils import save_mp4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--clips", nargs="+", required=True,
                   help="clip paths relative to data_root (clips/chunk_X/name.mp4)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_model", default="/nfs/zeqingwang/models/base_model")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--action_cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--action_context_table", default=None,
                   help="Enable per-frame action-context (table) mode.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pipe = SFPipeline.from_pretrained(
        base_model_dir=args.base_model,
        checkpoint_path=args.ckpt,
        variant="vrising",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    if args.action_context_table:
        pipe.load_action_context_table(args.action_context_table)

    for rel in args.clips:
        gt = os.path.join(args.data_root, rel)
        parq = os.path.join(args.data_root,
                            rel.replace("clips/", "actions/", 1)
                               .replace(".mp4", ".parquet"))
        stem = Path(rel).stem
        ff = os.path.join(args.out_dir, f"{stem}_f0.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", gt,
                        "-frames:v", "1", ff], check=True)
        out = pipe(
            image=Image.open(ff), actions_parquet=parq,
            num_frames=101, num_inference_steps=args.steps,
            cfg_scale=args.cfg, action_cfg_scale=args.action_cfg,
            seed=args.seed,
        )
        gen = os.path.join(args.out_dir, f"{stem}_gen.mp4")
        save_mp4(out.frames[0], gen, fps=20)
        cmp_path = os.path.join(args.out_dir, f"{stem}_cmp.mp4")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", gen, "-i", gt,
             "-filter_complex",
             "[0:v]drawtext=text='GEN':x=10:y=10:fontsize=28:fontcolor=yellow[l];"
             "[1:v]drawtext=text='GT':x=10:y=10:fontsize=28:fontcolor=lime[r];"
             "[l][r]hstack", "-c:v", "libx264", "-crf", "18", cmp_path],
            check=True)
        from vrising_data.verify import overlay_clip
        hud = cmp_path.replace(".mp4", "_hud.mp4")
        overlay_clip(cmp_path, parq, hud)
        print(f"[eval] {hud}", flush=True)


if __name__ == "__main__":
    main()
