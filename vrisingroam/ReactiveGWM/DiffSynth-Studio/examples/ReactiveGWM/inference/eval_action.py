"""Action-axis eval: profile.action_presets × fixed neutral strategy prompt.

Each job feeds a constant-pressed action across the whole 5s clip and renders
a video. Output per job:
  <out>/<job_id>.mp4
  <out>/<job_id>.json   (metrics + meta)
After all jobs finish, a report.md aggregates region motion energy by job_id.

The 13-action grid comes from `profile.action_presets` (see data/profiles.py),
so the SF2 / SF3 mapping difference (LP/MP/HP and LK/MK/HK on different physical
buttons) is encoded centrally — eval_action.py itself is game-agnostic.

Neutral prompt: in the original SF2/SF3 forks each game shipped a built-in
neutral string. ReactiveGWM keeps a profile-provided default but accepts
`--neutral_prompt` so per-game launch scripts can pin the *exact* training-time
text (which mostly differs only in the "Walk Left/Right" wording).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REACTIVE_GWM = _HERE.parent
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from data.profiles import get_profile, profile_names  # noqa: E402
from inference._mp_runner import Job, WorkerConfig, run_jobs  # noqa: E402


# Per-game neutral prompt. The strategy text is one of the v5 sub-strategies
# present in each game's training metadata; sentence form must match what the
# model saw at training time or me_* metrics drift artificially. These match
# the upstream SF2 / SF3 forks byte-for-byte.
NEUTRAL_PROMPTS = {
    "sf2": (
        "NPC: Active_Behavior(Walk Left: Moves character towards the left side of the screen.; "
        "Walk Right: Moves character towards the right side of the screen.), "
        "Passive_Behavior(Idle: Stands perfectly still without any input, waiting for an interaction.), "
        "Strategy(Defensive Spacing: Focuses on maintaining optimal distance using basic "
        "sweeping attacks and walking to intercept approaches.)"
    ),
    "sf3": (
        "NPC: Active_Behavior(Walk Left: Walking horizontally to the left side of the screen.; "
        "Walk Right: Walking horizontally to the right side of the screen.), "
        "Passive_Behavior(Idle: Stands still without input, waiting for an interaction.), "
        "Strategy(Control: Balances offense and defense by controlling distance, "
        "neither rushing in nor purely turtling.)"
    ),
}


def write_report(profile_name: str, out_dir: Path) -> None:
    rows = []
    for j in sorted(out_dir.glob("*.json")):
        if j.name in {"_run_args.json", "summary.json"}:
            continue
        rows.append(json.loads(j.read_text()))
    if not rows:
        print("[eval_action] no job outputs found.")
        return
    rows.sort(key=lambda r: r["job_id"])
    lines = [
        f"# ReactiveGWM ({profile_name}) — Action axis evaluation",
        "",
        "Each row is one constant-pressed action across a 5s clip with a fixed",
        "neutral Strategy prompt. `me_*` is mean abs frame-to-frame pixel diff",
        "(motion energy) in the named region (left = player half, right = NPC half).",
        "",
        "| job_id | me_full | me_left | me_right |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append(f"| {r['job_id']} | {m['me_full']:.3f} | {m['me_left']:.3f} | {m['me_right']:.3f} |")
    lines.append("")
    lines.append("**Sanity check**: me_left should be HIGHER for LEFT/RIGHT/UP_LEFT/UP_RIGHT")
    lines.append("than for noop. If they're all the same, the action conditioning isn't propagating.")
    (out_dir / "report.md").write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps({"jobs": rows}, indent=2, ensure_ascii=False))
    print(f"[eval_action] report.md + summary.json written to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="ReactiveGWM action-axis evaluation")
    p.add_argument("--game", default="sf2", choices=profile_names(),
                   help="Game profile (drives action presets and default neutral prompt).")
    p.add_argument("--full_ckpt", default=None,
                   help="Full DiT state dict (mode A/B1/B2 ckpt). Either this or "
                        "--lora_ckpt must be set.")
    p.add_argument("--lora_ckpt", default=None,
                   help="LoRA state dict (mode C ckpt).")
    p.add_argument("--lora_alpha", type=float, default=0.8)
    p.add_argument("--base_model_dir", default="/home/zeqingwang/zeqingwang/models/base_model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--reference_clip", required=True,
                   help="Path to a video.mp4 whose first frame is the input image for every job.")
    p.add_argument("--height", type=int, default=None,
                   help="Render height; defaults to profile.default_height.")
    p.add_argument("--width", type=int, default=None,
                   help="Render width; defaults to profile.default_width.")
    p.add_argument("--num_frames", type=int, default=None,
                   help="Defaults to profile.default_num_frames.")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="Prompt CFG. Set >1.0 only if the ckpt was trained with "
                        "--prompt_dropout_prob > 0.")
    p.add_argument("--action_cfg_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_ids", default="0",
                   help="Comma-separated GPU ids (e.g. '0,1,2,3').")
    p.add_argument("--neutral_prompt", default=None,
                   help="Override the per-game default neutral prompt. Useful when "
                        "matching a specific training metadata's strategy text exactly.")
    p.add_argument("--report_only", action="store_true",
                   help="Skip inference, just regenerate report.md from existing JSONs.")
    args = p.parse_args()

    profile = get_profile(args.game)
    height = args.height if args.height is not None else profile.default_height
    width = args.width if args.width is not None else profile.default_width
    num_frames = args.num_frames if args.num_frames is not None else profile.default_num_frames
    neutral_prompt = args.neutral_prompt or NEUTRAL_PROMPTS.get(args.game) or profile.fixed_prompt

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_run_args.json").write_text(json.dumps(
        {**vars(args), "_resolved": {"height": height, "width": width,
                                     "num_frames": num_frames,
                                     "neutral_prompt": neutral_prompt}},
        indent=2, ensure_ascii=False))

    if args.report_only:
        write_report(profile.name, out_dir)
        return

    jobs: list[Job] = []
    for slug, btns in profile.action_presets:
        jobs.append(Job(
            job_id=slug,
            prompt=neutral_prompt,
            action_spec={"kind": "constant", "buttons": list(btns)},
            input_image_path=args.reference_clip,
            seed=args.seed,
            extra={"action_label": slug, "buttons": list(btns)},
        ))

    cfg = WorkerConfig(
        base_model_dir=args.base_model_dir,
        full_ckpt=args.full_ckpt, lora_ckpt=args.lora_ckpt,
        lora_alpha=args.lora_alpha,
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale, action_cfg_scale=args.action_cfg_scale,
        output_dir=str(out_dir),
        game=profile.name,
    )
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    print(f"[eval_action] game={profile.name} {len(jobs)} jobs across GPUs {gpu_ids}")
    run_jobs(jobs, cfg, gpu_ids, str(_REACTIVE_GWM))
    write_report(profile.name, out_dir)


if __name__ == "__main__":
    main()
