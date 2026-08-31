"""SF2_final focus eval: 6 in-distribution prompts × 2 first frames × 2 seeds = 24 jobs.

Goal: test whether the model follows the Strategy slot when Active_Behavior also
matches the training distribution (e.g. Offense prompts with attack actions).

Reads job spec from /tmp/sf2_focus_eval_spec.json (built externally).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REACTIVE_GWM = _HERE.parents[1]  # examples/ReactiveGWM/
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from inference._mp_runner import Job, WorkerConfig, run_jobs  # noqa: E402


CLIPS_ROOT = Path("/home/zeqingwang/zeqingwang/datasets/Final_dataset/SF2/train/clips_5s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full_ckpt", default=None)
    p.add_argument("--lora_ckpt", default=None)
    p.add_argument("--lora_alpha", type=float, default=0.8)
    p.add_argument("--base_model_dir", default="/home/zeqingwang/zeqingwang/models/base_model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--spec", default="/tmp/sf2_focus_eval_spec.json")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=608)
    p.add_argument("--num_frames", type=int, default=101)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--action_cfg_scale", type=float, default=1.0)
    p.add_argument("--gpu_ids", default="0")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(args.spec).read_text())
    refs = [("ref1", spec["ref1"]), ("ref2", spec["ref2"])]
    seeds = [0, 1]

    jobs: list[Job] = []
    for entry in spec["prompts"]:
        for ref_name, rel in refs:
            for seed in seeds:
                jid = f"{entry['slug']}__{ref_name}__seed{seed}"
                jobs.append(Job(
                    job_id=jid,
                    prompt=entry["prompt"],
                    action_spec={"kind": "segments", "segments": [[], [], [], []]},
                    input_image_path=str(CLIPS_ROOT / rel),
                    seed=seed,
                    extra={
                        "category": entry["category"],
                        "slug": entry["slug"],
                        "ref": ref_name,
                        "ref_video": rel,
                        "active_names": entry["active_names"],
                    },
                ))

    cfg = WorkerConfig(
        base_model_dir=args.base_model_dir,
        full_ckpt=args.full_ckpt, lora_ckpt=args.lora_ckpt,
        lora_alpha=args.lora_alpha,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale, action_cfg_scale=args.action_cfg_scale,
        output_dir=str(out_dir),
        game="sf2",
    )
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    print(f"[eval_focus] {len(jobs)} jobs across GPUs {gpu_ids}")
    print(f"[eval_focus] cfg_scale={args.cfg_scale}  action_cfg_scale={args.action_cfg_scale}")
    run_jobs(jobs, cfg, gpu_ids, str(_REACTIVE_GWM))
    print(f"[eval_focus] done — outputs in {out_dir}")


if __name__ == "__main__":
    main()
