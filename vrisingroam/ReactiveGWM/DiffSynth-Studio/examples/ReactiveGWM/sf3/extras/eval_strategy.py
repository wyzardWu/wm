"""SF3 strategy-axis eval: 9 v5 sub-strategies × shared action sequences.

The 9 sub-strategies below are byte-identical to the strings in the SF3 v5cat
training metadata (`metadata_wo_pure_v5cat_10k.csv`) — verified by reading the
file's unique `strategy` values. They reuse the SF2_final v5 taxonomy exactly,
NOT the SF3-native 9-tag taxonomy from `taxonomy.json` (Aggressive Rushdown /
Aerial Assault / etc.) which the v5cat training CSV does not actually use.

Each job uses a strategy-specific prompt and one of three shared action presets
(neutral, pressure, retreat). Output per job:
  <out>/<job_id>.mp4
  <out>/<job_id>.json
After all jobs finish, report.md aggregates NPC-half motion energy by strategy.
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


# v5 taxonomy: 9 sub-strategies = 3 categories × 3 descriptions each.
# Slug -> (category, description) — text matches metadata_wo_pure_v5cat_10k.csv.
SUB_STRATEGIES = {
    "defense_hold_ground": (
        "Defense",
        "Holds ground with blocks and reactive counters, only striking when an opening appears.",
    ),
    "defense_guard_read": (
        "Defense",
        "Prioritizes guarding and reading the opponent's actions over initiating offense.",
    ),
    "defense_absorb_evade": (
        "Defense",
        "Absorbs and evades incoming pressure, recovering safely instead of trading hits.",
    ),
    "control_spacing_projectiles": (
        "Control",
        "Manages spacing with projectiles and measured pokes to dictate the pace of engagement.",
    ),
    "control_balance_distance": (
        "Control",
        "Balances offense and defense by controlling distance, neither rushing in nor purely turtling.",
    ),
    "control_zoning_range": (
        "Control",
        "Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.",
    ),
    "offense_constant_aggression": (
        "Offense",
        "Maintains constant aggression to overwhelm the opponent and force defensive reactions.",
    ),
    "offense_close_distance": (
        "Offense",
        "Closes the distance quickly to apply pressure and initiate close combat.",
    ),
    "offense_advance_chain": (
        "Offense",
        "Focuses on advancing and chaining attacks to keep the opponent on the back foot.",
    ),
}


def make_strategy_prompt(slug: str) -> str:
    """Build a minimal prompt that fixes the Strategy slot only.

    Active/Passive behavior slots are left as a generic neutral pair so the
    only varying signal between jobs is the strategy declaration. The Active /
    Passive vocabulary is drawn from what training data uses (Idle / Walk Left
    / Walk Right with the exact descriptions seen in v5cat CSV).
    """
    cat, desc = SUB_STRATEGIES[slug]
    return (
        "NPC: Active_Behavior(Walk Left: Walking horizontally to the left side of the screen.; "
        "Walk Right: Walking horizontally to the right side of the screen.), "
        "Passive_Behavior(Idle: Stands still without input, waiting for an interaction.), "
        f"Strategy({cat}: {desc})"
    )


# Three shared action presets (4-segment sequences across num_frames).
# ⚠ SF3 button-strength mapping: A=light/Y=medium/X=heavy punch,
#                                B=light/C=medium/Z=heavy kick.
# `pressure` = approach + light punch + approach + heavy punch.
ACTION_PRESETS = {
    "neutral":  {"kind": "segments", "segments": [[], [], [], []]},
    "pressure": {"kind": "segments",
                 "segments": [["LEFT"], ["A"], ["LEFT"], ["X"]]},
    "retreat":  {"kind": "segments",
                 "segments": [["RIGHT"], ["RIGHT"], [], ["DOWN"]]},
}


def write_report(out_dir: Path) -> None:
    rows = []
    for j in sorted(out_dir.glob("*.json")):
        if j.name == "_run_args.json":
            continue
        rows.append(json.loads(j.read_text()))
    if not rows:
        print("[eval_strategy] no job outputs found.")
        return
    rows.sort(key=lambda r: r["job_id"])
    lines = [
        "# SF3 — Strategy axis evaluation",
        "",
        "Rows are (sub-strategy × action preset). me_right is the NPC-half motion",
        "energy — the strategy axis should produce a clear me_right gradient",
        "(Offense > Control > Defense).",
        "",
        "| job_id | category | strategy | action_preset | me_full | me_left | me_right |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        m = r["metrics"]
        e = r["extra"]
        lines.append(
            f"| {r['job_id']} | {e['category']} | {e['strategy']} | "
            f"{e['action_preset']} | {m['me_full']:.3f} | {m['me_left']:.3f} | "
            f"{m['me_right']:.3f} |"
        )
    lines.append("")
    lines.append("**Sanity check**: me_right should differ across strategy at fixed action_preset.")
    lines.append("If me_right is flat across strategies, the prompt isn't reaching cross_attn.")
    (out_dir / "report.md").write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps({"jobs": rows}, indent=2, ensure_ascii=False))
    print(f"[eval_strategy] report.md + summary.json written to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="SF3 strategy-axis evaluation")
    p.add_argument("--full_ckpt", default=None)
    p.add_argument("--lora_ckpt", default=None)
    p.add_argument("--lora_alpha", type=float, default=0.8)
    p.add_argument("--base_model_dir",
                   default="/opt/dlami/nvme/zeqingwang/models/base_model")
    p.add_argument("--output_dir", required=True,
                   help="Recommended layout: /opt/dlami/nvme/zeqingwang/evaluation/sf3/"
                        "<run_name>_step<step>/strategy/")
    p.add_argument("--reference_clip", required=True,
                   help="Path to one source clip's video.mp4 — its first frame is the "
                        "shared input image across all 27 jobs.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=101)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="Prompt CFG. Set >1.0 (e.g. 5.0) only if the ckpt was trained "
                        "with --prompt_dropout_prob > 0.")
    p.add_argument("--action_cfg_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--report_only", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_run_args.json").write_text(json.dumps(vars(args), indent=2))

    if args.report_only:
        write_report(out_dir)
        return

    jobs: list[Job] = []
    for slug, (cat, desc) in SUB_STRATEGIES.items():
        for preset_name, action_spec in ACTION_PRESETS.items():
            jobs.append(Job(
                job_id=f"{slug}__{preset_name}",
                prompt=make_strategy_prompt(slug),
                action_spec=action_spec,
                input_image_path=args.reference_clip,
                seed=args.seed,
                extra={
                    "strategy": f"{cat}: {desc}",
                    "category": cat,
                    "action_preset": preset_name,
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
        game="sf3",
    )
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    print(f"[eval_strategy] {len(jobs)} jobs across GPUs {gpu_ids}")
    run_jobs(jobs, cfg, gpu_ids, str(_REACTIVE_GWM))
    write_report(out_dir)


if __name__ == "__main__":
    main()
