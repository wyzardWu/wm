"""SF3 strategy eval with SF2-style minimal prompts.

Same 6 strategy descriptions as eval_strategy_6.py, but wrapped in SF2's
generic Active_Behavior(Walk Left/Walk Right) + Passive_Behavior(Idle) shell
instead of category-specific behaviors. This isolates the cross_attn response
to the Strategy(...) slot alone, useful when probing a cross_attn taken from
an SF2-trained model that hasn't seen SF3 active-behavior vocabulary.
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

# 6 strategy descriptions reused byte-exact from SF3 eval_strategy_6.SUB_STRATEGIES,
# but rebuilt using SF2's minimal prompt format. Slug -> (category, description).
SUB_STRATEGIES = {
    "offense_punch_chain": (
        "Offense",
        "Focuses on advancing and chaining attacks to keep the opponent on the back foot.",
    ),
    "offense_jump_in_attack": (
        "Offense",
        "Closes the distance quickly to apply pressure and initiate close combat.",
    ),
    "defense_crouch_block": (
        "Defense",
        "Holds ground with blocks and reactive counters, only striking when an opening appears.",
    ),
    "defense_absorb_evade": (
        "Defense",
        "Absorbs and evades incoming pressure, recovering safely instead of trading hits.",
    ),
    "control_walk_spacing": (
        "Control",
        "Manages spacing with projectiles and measured pokes to dictate the pace of engagement.",
    ),
    "control_zoning_kunai": (
        "Control",
        "Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.",
    ),
}


def make_strategy_prompt(cat: str, desc: str) -> str:
    """SF2-style prompt: minimal Active/Passive shell + Strategy slot."""
    return (
        "NPC: Active_Behavior(Walk Left: Moves character towards the left side of the screen.; "
        "Walk Right: Moves character towards the right side of the screen.), "
        "Passive_Behavior(Idle: Stands perfectly still without any input, waiting for an interaction.), "
        f"Strategy({cat}: {desc})"
    )


NEUTRAL_PRESET = {"kind": "segments", "segments": [[], [], [], []]}


def write_report(out_dir: Path) -> None:
    rows = []
    for j in sorted(out_dir.glob("*.json")):
        if j.name in ("_run_args.json", "summary.json"):
            continue
        rows.append(json.loads(j.read_text()))
    if not rows:
        print("[eval_strategy_6_sf2prompt] no job outputs found.")
        return
    rows.sort(key=lambda r: (r["extra"]["category"], r["job_id"]))
    lines = [
        "# SF3 — Strategy axis evaluation (6 jobs, SF2-style minimal prompts)",
        "",
        "Same 6 strategy descriptions as eval_strategy_6 but wrapped in SF2's",
        "generic Active(Walk L/R) + Passive(Idle) shell. Only the Strategy(...)",
        "slot varies — designed to isolate cross_attn response when the cross",
        "weights come from an SF2-trained model.",
        "",
        "| job_id | category | strategy | me_full | me_left | me_right |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        m = r["metrics"]
        e = r["extra"]
        lines.append(
            f"| {r['job_id']} | {e['category']} | {e['strategy']} | "
            f"{m['me_full']:.3f} | {m['me_left']:.3f} | {m['me_right']:.3f} |"
        )
    lines.append("")
    lines.append("Sanity: me_right should differ across categories.")
    (out_dir / "report.md").write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps({"jobs": rows}, indent=2, ensure_ascii=False))
    print(f"[eval_strategy_6_sf2prompt] report.md + summary.json -> {out_dir}")


def main():
    p = argparse.ArgumentParser(description="SF3 strategy eval with SF2-style minimal prompts (6 jobs)")
    p.add_argument("--full_ckpt", default=None)
    p.add_argument("--lora_ckpt", default=None)
    p.add_argument("--lora_alpha", type=float, default=0.8)
    p.add_argument("--base_model_dir", default="/home/zeqingwang/zeqingwang/models/base_model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--reference_clip", required=True)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=101)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--cfg_scale", type=float, default=1.0)
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
        jobs.append(Job(
            job_id=slug,
            prompt=make_strategy_prompt(cat, desc),
            action_spec=NEUTRAL_PRESET,
            input_image_path=args.reference_clip,
            seed=args.seed,
            extra={
                "strategy": f"{cat}: {desc}",
                "category": cat,
                "action_preset": "neutral",
                "prompt_style": "sf2_minimal",
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
    print(f"[eval_strategy_6_sf2prompt] {len(jobs)} jobs across GPUs {gpu_ids}")
    run_jobs(jobs, cfg, gpu_ids, str(_REACTIVE_GWM))
    write_report(out_dir)


if __name__ == "__main__":
    main()
