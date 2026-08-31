"""SF2_ALL baseline — 3-model × strategy-prompt evaluation.

Matrix:
  3 models × 5 first-frames × 6 prompts × 3 seeds = 270 jobs
  prompts: 2 per category (Defense / Control / Offense), full structured prompts
           pulled from real metadata_v5.csv rows
  action:  empty keyboard (10×101 zeros) — pure prompt-conditioning probe
  params:  480×608, 101 frames, 30 steps, cfg=5.0, action_cfg=1.0

Layout:
  <out_dir>/prompts.json                      (already written by pick step)
  <out_dir>/jobs.csv                          (per-job metadata, written here)
  <out_dir>/<model>/<job_id>.mp4              (+ <job_id>.json)
  <out_dir>/report.md                         (aggregate me_left/right after all jobs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REACTIVE_GWM = _HERE.parents[1]  # examples/ReactiveGWM/
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from inference._mp_runner import Job, WorkerConfig, run_jobs  # noqa: E402


# --- fixed eval matrix ---------------------------------------------------------------

# 5 first-frame source clips (idx 00, 01, 04, 06, 07 from picker grid).
CLIPS = [
    "random/clip_0451726/video.mp4",
    "random/clip_0054414/video.mp4",
    "random/clip_0000235/video.mp4",
    "random/clip_0325169/video.mp4",
    "random/clip_0090700/video.mp4",
]

# 3 model checkpoints (all loaded as full_ckpt via build_pipe + load_state_dict).
MODELS = [
    {
        "name": "base",
        "ckpt": "/home/zeqingwang/zeqingwang/models/Baseline/SF2_ALL/base/step-39000.safetensors",
    },
    {
        "name": "visual",
        "ckpt": "/home/zeqingwang/zeqingwang/models/Baseline/SF2_ALL/visual/step-40000.safetensors",
    },
    {
        "name": "visual_xattn_sf3",
        "ckpt": ("/home/zeqingwang/zeqingwang/models/Baseline/SF2_ALL/visual_xattn_sf3/"
                 "visual_step40000__xattn_SF3_step44000.safetensors"),
    },
]

SEEDS = [0, 42, 1234]
DATASET_ROOT = Path("/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s")

# Empty-action spec: 10 buttons × 101 frames all zero.  Encoded for `_mp_runner.build_action_tensor`.
EMPTY_ACTION = {"kind": "constant", "buttons": []}


def build_job_list(prompts: list[dict], output_dir: Path) -> list[Job]:
    """5 clips × 6 prompts × 3 seeds = 90 jobs (per model)."""
    jobs: list[Job] = []
    for clip_rel in CLIPS:
        clip_id = clip_rel.split("/")[1]                         # clip_0451726
        first_frame = str(DATASET_ROOT / clip_rel)
        for p in prompts:
            cat = p["category"]
            pid = p["prompt_idx"]
            for seed in SEEDS:
                job_id = f"{clip_id}__{cat}_p{pid}__seed{seed}"
                jobs.append(Job(
                    job_id=job_id,
                    prompt=p["prompt"],
                    action_spec=EMPTY_ACTION,
                    input_image_path=first_frame,
                    seed=seed,
                    extra={
                        "clip_id": clip_id,
                        "category": cat,
                        "prompt_idx": pid,
                        "sub_strategy": p["sub_strategy"],
                        "src_clip_for_prompt": p.get("src_clip", ""),
                    },
                ))
    return jobs


def write_jobs_csv(jobs: list[Job], out_path: Path) -> None:
    rows = []
    for j in jobs:
        rows.append({
            "job_id":   j.job_id,
            "clip_id":  j.extra["clip_id"],
            "first_frame": j.input_image_path,
            "category": j.extra["category"],
            "prompt_idx": j.extra["prompt_idx"],
            "seed":     j.seed,
            "sub_strategy": j.extra["sub_strategy"],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_report(out_dir: Path) -> None:
    """Aggregate per-job metrics across all 3 models into report.md.

    Each per-job json (written by _mp_runner) has metrics.me_full/left/right.
    """
    rows = []
    for model in MODELS:
        mdir = out_dir / model["name"]
        if not mdir.is_dir():
            continue
        for jp in sorted(mdir.glob("*.json")):
            try:
                d = json.loads(jp.read_text())
            except Exception:
                continue
            m = d.get("metrics", {}); e = d.get("extra", {})
            rows.append({
                "model": model["name"],
                "job_id": d.get("job_id"),
                "clip_id": e.get("clip_id"),
                "category": e.get("category"),
                "prompt_idx": e.get("prompt_idx"),
                "seed": d.get("seed"),
                "me_full": m.get("me_full"),
                "me_left": m.get("me_left"),
                "me_right": m.get("me_right"),
            })
    if not rows:
        print("[report] no per-job .json found yet; skipping report.md")
        return
    df = pd.DataFrame(rows)
    summary_csv = out_dir / "summary.csv"
    df.to_csv(summary_csv, index=False)

    lines = [
        "# SF2_ALL baseline — strategy evaluation",
        "",
        f"Total jobs: {len(df)}  ·  models: {df['model'].nunique()}  ·  ",
        f"first-frames: {df['clip_id'].nunique()}  ·  prompts: 6  ·  seeds: {df['seed'].nunique()}",
        "",
        "## Mean motion energy (NPC = right half) by model × category",
        "",
        "| model | category | me_full | me_left | me_right |",
        "|---|---|---:|---:|---:|",
    ]
    g = (df.groupby(["model", "category"])[["me_full", "me_left", "me_right"]]
            .mean().reset_index().sort_values(["model", "category"]))
    for _, r in g.iterrows():
        lines.append(f"| {r['model']} | {r['category']} | "
                     f"{r['me_full']:.3f} | {r['me_left']:.3f} | {r['me_right']:.3f} |")

    lines += [
        "",
        "## Sanity checks",
        "- For each model, `me_right` should differ across category (Offense > Control > Defense ideally).",
        "- Across models on the same (clip_id, category, prompt_idx, seed), motion energy difference",
        "  reflects the architectural difference (base vs visual vs visual_xattn_sf3).",
        "",
        f"Per-job metrics dumped to `summary.csv` ({summary_csv.name}).",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"[report] wrote {out_dir/'report.md'} and {summary_csv}")


def main():
    p = argparse.ArgumentParser(description="SF2_ALL 3-model strategy evaluation")
    p.add_argument("--output_dir", default="/home/zeqingwang/zeqingwang/Paper_Figure/Strategy/SF2")
    p.add_argument("--prompts_json", default=None,
                   help="defaults to <output_dir>/prompts.json")
    p.add_argument("--base_model_dir", default="/home/zeqingwang/zeqingwang/models/base_model")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width",  type=int, default=608)
    p.add_argument("--num_frames", type=int, default=101)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--action_cfg_scale", type=float, default=1.0)
    p.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7")
    p.add_argument("--report_only", action="store_true",
                   help="skip inference, only re-aggregate report.md from existing .json")
    p.add_argument("--only_model", default=None,
                   help="comma-list of model names to run; default = all 3")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        write_report(out_dir)
        return

    prompts_json = Path(args.prompts_json) if args.prompts_json else out_dir / "prompts.json"
    if not prompts_json.exists():
        raise SystemExit(f"prompts.json not found at {prompts_json}; run pick step first.")
    prompts = json.loads(prompts_json.read_text())
    print(f"[eval] loaded {len(prompts)} prompts from {prompts_json}")

    jobs_template = build_job_list(prompts, out_dir)
    print(f"[eval] {len(jobs_template)} jobs/model × {len(MODELS)} models = "
          f"{len(jobs_template) * len(MODELS)} total")

    write_jobs_csv(jobs_template, out_dir / "jobs.csv")
    print(f"[eval] wrote {out_dir/'jobs.csv'}")

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    print(f"[eval] using GPUs {gpu_ids}")

    only_models = set(args.only_model.split(",")) if args.only_model else None

    for model in MODELS:
        if only_models is not None and model["name"] not in only_models:
            continue
        mdir = out_dir / model["name"]
        mdir.mkdir(parents=True, exist_ok=True)
        ckpt = model["ckpt"]
        if not Path(ckpt).is_file():
            raise SystemExit(f"ckpt missing: {ckpt}")

        cfg = WorkerConfig(
            base_model_dir=args.base_model_dir,
            full_ckpt=ckpt, lora_ckpt=None, lora_alpha=0.8,
            height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale, action_cfg_scale=args.action_cfg_scale,
            output_dir=str(mdir),
            game="sf2",
        )

        # Skip already-finished jobs in this model dir.
        jobs = [j for j in jobs_template if not (mdir / f"{j.job_id}.json").exists()]
        skipped = len(jobs_template) - len(jobs)
        if skipped:
            print(f"[eval] [{model['name']}] resuming: {skipped} already done, "
                  f"{len(jobs)} pending")

        if not jobs:
            print(f"[eval] [{model['name']}] all jobs already done — skipping")
            continue

        print(f"[eval] [{model['name']}] launching {len(jobs)} jobs across {len(gpu_ids)} GPUs")
        run_jobs(jobs, cfg, gpu_ids, str(_REACTIVE_GWM))
        print(f"[eval] [{model['name']}] DONE\n")

    write_report(out_dir)


if __name__ == "__main__":
    main()
