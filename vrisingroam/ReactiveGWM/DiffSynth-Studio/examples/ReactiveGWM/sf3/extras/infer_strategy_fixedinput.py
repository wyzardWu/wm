"""Strategy listening eval with fixed input image and fixed (zero) action.

Runs N strategy prompts against a single reference clip (its first frame as
input image) and a zero keyboard action across all frames. Designed for
single-GPU use; bash launcher pins CUDA_VISIBLE_DEVICES before invoking python
and shards strategies across processes.

Output filenames are `<slug>.mp4` (slug from samples.csv -> STRATEGY_SLUGS).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path("/home/zeqingwang/zeqingwang/Training_Wan")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_REACTIVE_GWM = Path(__file__).resolve().parents[1]  # examples/ReactiveGWM/
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from inference.infer_core import (  # noqa: E402
    CkptSpec, build_pipe, first_frame_pil, run_inference,
)
from data.profiles import SF3 as _PROFILE
SF3_BUTTON_COLS = list(_PROFILE.button_cols)  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402


# Strategy text -> slug. Keys must match metadata_wo_pure_v5cat_10k.csv exactly.
STRATEGY_SLUGS = {
    "Offense: Closes the distance quickly to apply pressure and initiate close combat.":
        "offense_close_distance",
    "Offense: Maintains constant aggression to overwhelm the opponent and force defensive reactions.":
        "offense_constant_aggression",
    "Offense: Focuses on advancing and chaining attacks to keep the opponent on the back foot.":
        "offense_advance_chain",
    "Defense: Holds ground with blocks and reactive counters, only striking when an opening appears.":
        "defense_hold_ground",
    "Defense: Prioritizes guarding and reading the opponent's actions over initiating offense.":
        "defense_guard_read",
    "Defense: Absorbs and evades incoming pressure, recovering safely instead of trading hits.":
        "defense_absorb_evade",
    "Control: Manages spacing with projectiles and measured pokes to dictate the pace of engagement.":
        "control_spacing_projectiles",
    "Control: Balances offense and defense by controlling distance, neither rushing in nor purely turtling.":
        "control_balance_distance",
    "Control: Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.":
        "control_zoning_range",
}


def zero_action(num_frames: int, device: str = "cuda") -> torch.Tensor:
    arr = np.zeros((num_frames, len(SF3_BUTTON_COLS)), dtype=np.float32)
    return torch.tensor(arr).unsqueeze(0).to(device, torch.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--base_model_dir", default="/home/zeqingwang/zeqingwang/models/base_model")
    p.add_argument("--reference_video", required=True,
                   help="mp4 whose first frame becomes the shared input image")
    p.add_argument("--samples_csv", required=True,
                   help="CSV with columns video, prompt, strategy, category — "
                        "used to look up the prompt text for each requested slug.")
    p.add_argument("--slugs", required=True,
                   help="comma-separated list of strategy slugs to run on this process")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=101)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--action_cfg_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_slugs = [s for s in args.slugs.split(",") if s.strip()]
    df = pd.read_csv(args.samples_csv)
    slug_to_row = {STRATEGY_SLUGS.get(r["strategy"].strip()): r for _, r in df.iterrows()}
    jobs = []
    for slug in requested_slugs:
        if slug not in slug_to_row:
            raise SystemExit(f"slug {slug!r} not found in samples_csv")
        row = slug_to_row[slug]
        jobs.append({
            "slug": slug,
            "prompt": row["prompt"],
            "strategy": row["strategy"],
            "category": row["category"],
        })

    print(f"[fixedinput] {len(jobs)} jobs on this process, ref={args.reference_video}")

    pipe = build_pipe(args.base_model_dir, CkptSpec(full_ckpt=args.full_ckpt), device="cuda")
    input_image = first_frame_pil(args.reference_video, args.height, args.width)
    keyboard = zero_action(args.num_frames, device="cuda")

    for i, job in enumerate(jobs):
        out_mp4 = out_dir / f"{job['slug']}.mp4"
        if out_mp4.exists():
            print(f"[fixedinput] skip (exists): {job['slug']}")
            continue
        try:
            video = run_inference(
                pipe=pipe,
                prompt=job["prompt"],
                keyboard_action=keyboard,
                input_image=input_image,
                num_frames=args.num_frames,
                height=args.height, width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                action_cfg_scale=args.action_cfg_scale,
                seed=args.seed,
            )
            save_video(video, str(out_mp4), fps=20, quality=5)
            (out_dir / f"{job['slug']}.json").write_text(json.dumps({
                "slug": job["slug"],
                "category": job["category"],
                "strategy": job["strategy"],
                "prompt": job["prompt"],
                "reference_video": str(args.reference_video),
                "action": "zero",
                "cfg_scale": args.cfg_scale,
                "action_cfg_scale": args.action_cfg_scale,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "height": args.height, "width": args.width,
                "seed": args.seed,
                "ckpt": args.full_ckpt,
            }, indent=2, ensure_ascii=False))
            print(f"[fixedinput] {i+1}/{len(jobs)} done: {job['slug']}")
        except Exception as e:  # noqa: BLE001
            (out_dir / f"{job['slug']}.error.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
            print(f"[fixedinput] FAIL {job['slug']}: {e}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
