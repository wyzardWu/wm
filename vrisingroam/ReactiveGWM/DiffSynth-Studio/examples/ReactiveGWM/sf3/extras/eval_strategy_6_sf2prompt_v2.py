"""SF3 strategy eval with SF2-style verbose prompts (category-aligned active behaviors).

v2: Active_Behavior is no longer the generic Walk L/R shell. Each prompt is
copy-pasted byte-exact from SF2 v5 training metadata so that the active behavior
list aligns with the strategy category (Offense -> Standing Punch / Standing
Kick / Jumping Attack; Defense -> Crouch / Block; Control -> Sonic Boom / Crouch).

This isolates "what SF2 cross_attn does when shown SF2-vocabulary active
behaviors" without falling back to generic Walk L/R that strips offense
signal entirely.
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

# Byte-exact prompts pulled from SF2 metadata_v5_pure.csv. Same 6 strategy
# descriptions as eval_strategy_6.py / eval_strategy_6_sf2prompt.py, but the
# surrounding Active_Behavior list is SF2-vocabulary and category-aligned.
SUB_STRATEGIES = {
    "offense_punch_chain": (
        "Offense",
        "Focuses on advancing and chaining attacks to keep the opponent on the back foot.",
        "NPC: Active_Behavior(Walk Left: Moves character towards the left side of the screen.; Jump Left: Executes an aerial jump towards the left side of the screen.; Standing Punch: Executes a basic punch attack while in a standing posture.; Standing Kick: Executes a basic kick attack while in a standing posture.; Jumping Attack: Executes a punch or kick attack while airborne.), Passive_Behavior(Standing Block: Passively guards against high or mid attacks while standing.; Take Damage: Absorbs a hit from an opponent's attack, resulting in hitstun and health loss.; Knockback: Pushed away by the force of an opponent's blocked or landed attack.), Strategy(Offense: Focuses on advancing and chaining attacks to keep the opponent on the back foot.)",
    ),
    "offense_jump_in_attack": (
        "Offense",
        "Closes the distance quickly to apply pressure and initiate close combat.",
        "NPC: Active_Behavior(Walk Left: Moves character towards the left side of the screen.; Standing Punch: Executes a basic punch attack while in a standing posture.; Standing Kick: Executes a basic kick attack while in a standing posture.; Crouching Kick: Executes a low sweep or kick attack from a crouching posture.; Crouch: Enters and holds a crouching stance to lower the character's hitbox and prepare charged moves.), Passive_Behavior(N/A), Strategy(Offense: Closes the distance quickly to apply pressure and initiate close combat.)",
    ),
    "defense_crouch_block": (
        "Defense",
        "Holds ground with blocks and reactive counters, only striking when an opening appears.",
        "NPC: Active_Behavior(Crouch: Enters and holds a crouching stance to lower the character's hitbox and prepare charged moves.), Passive_Behavior(Idle: Stands perfectly still without any input, waiting for an interaction.; Crouching Block: Passively guards against low and mid attacks from a lowered stance.; Take Damage: Absorbs a hit from an opponent's attack, resulting in hitstun and health loss.; Knockback: Pushed away by the force of an opponent's blocked or landed attack.), Strategy(Defense: Holds ground with blocks and reactive counters, only striking when an opening appears.)",
    ),
    "defense_absorb_evade": (
        "Defense",
        "Absorbs and evades incoming pressure, recovering safely instead of trading hits.",
        "NPC: Active_Behavior(Walk Right: Moves character towards the right side of the screen.; Crouch: Enters and holds a crouching stance to lower the character's hitbox and prepare charged moves.), Passive_Behavior(Crouching Block: Passively guards against low and mid attacks from a lowered stance.; Standing Block: Passively guards against high or mid attacks while standing.), Strategy(Defense: Absorbs and evades incoming pressure, recovering safely instead of trading hits.)",
    ),
    "control_walk_spacing": (
        "Control",
        "Manages spacing with projectiles and measured pokes to dictate the pace of engagement.",
        "NPC: Active_Behavior(Jump Left: Executes an aerial jump towards the left side of the screen.; Jump In Place: Executes a neutral vertical jump into the air without horizontal movement.; Crouch: Enters and holds a crouching stance to lower the character's hitbox and prepare charged moves.; Sonic Boom: Guile's signature projectile attack thrown horizontally across the screen.), Passive_Behavior(Idle: Stands perfectly still without any input, waiting for an interaction.), Strategy(Control: Manages spacing with projectiles and measured pokes to dictate the pace of engagement.)",
    ),
    "control_zoning_kunai": (
        "Control",
        "Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.",
        "NPC: Active_Behavior(Crouch: Enters and holds a crouching stance to lower the character's hitbox and prepare charged moves.; Sonic Boom: Guile's signature projectile attack thrown horizontally across the screen.), Passive_Behavior(Crouching Block: Passively guards against low and mid attacks from a lowered stance.; Evade: Passively avoids an incoming attack or projectile.), Strategy(Control: Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.)",
    ),
}


NEUTRAL_PRESET = {"kind": "segments", "segments": [[], [], [], []]}


def write_report(out_dir: Path) -> None:
    rows = []
    for j in sorted(out_dir.glob("*.json")):
        if j.name in ("_run_args.json", "summary.json"):
            continue
        rows.append(json.loads(j.read_text()))
    if not rows:
        print("[eval_strategy_6_sf2prompt_v2] no job outputs found.")
        return
    rows.sort(key=lambda r: (r["extra"]["category"], r["job_id"]))
    lines = [
        "# SF3 — Strategy axis evaluation (6 jobs, SF2-vocabulary verbose prompts)",
        "",
        "Same 6 strategy descriptions as eval_strategy_6 but the entire prompt is",
        "byte-exact from SF2 metadata_v5_pure.csv — Active_Behavior list uses SF2",
        "vocabulary AND aligns with the strategy category (Offense -> Standing Punch /",
        "Standing Kick / Jumping Attack; Defense -> Crouch / Block; Control -> Sonic Boom).",
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
    print(f"[eval_strategy_6_sf2prompt_v2] report.md + summary.json -> {out_dir}")


def main():
    p = argparse.ArgumentParser(description="SF3 strategy eval w/ SF2-vocabulary prompts (6 jobs)")
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
    for slug, (cat, desc, full_prompt) in SUB_STRATEGIES.items():
        jobs.append(Job(
            job_id=slug,
            prompt=full_prompt,
            action_spec=NEUTRAL_PRESET,
            input_image_path=args.reference_clip,
            seed=args.seed,
            extra={
                "strategy": f"{cat}: {desc}",
                "category": cat,
                "action_preset": "neutral",
                "prompt_style": "sf2_verbose_aligned",
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
    print(f"[eval_strategy_6_sf2prompt_v2] {len(jobs)} jobs across GPUs {gpu_ids}")
    run_jobs(jobs, cfg, gpu_ids, str(_REACTIVE_GWM))
    write_report(out_dir)


if __name__ == "__main__":
    main()
