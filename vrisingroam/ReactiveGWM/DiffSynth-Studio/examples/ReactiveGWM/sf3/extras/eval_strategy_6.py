"""SF3 strategy-axis eval — slim 6-job version (2 prompts × 3 categories × 1 neutral preset).

Output:
  <out>/<job_id>.mp4
  <out>/<job_id>.json
  <out>/report.md
  <out>/summary.json

Prompts are byte-exact from the v5cat training metadata
(metadata_wo_pure_v5cat_10k.csv), so cfg>1 stays in-distribution.
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

# 6 prompts pulled byte-exact from metadata_wo_pure_v5cat_10k.csv. Each was
# picked so that its Active_Behavior matches the Strategy category semantically
# (Offense -> attacks, Defense -> Crouch/blocks, Control -> spacing/projectiles).
# slug -> (category, source_video, strategy, full_prompt)
SUB_STRATEGIES = {
    "offense_punch_chain": (
        "Offense",
        "random/clip_0098026/video.mp4",
        "Offense: Focuses on advancing and chaining attacks to keep the opponent on the back foot.",
        "NPC: Active_Behavior(Standing Punch: A standard hand strike (LP/MP/HP) thrown while standing.; Standing Kick: A standard foot strike (LK/MK/HK) thrown while standing.; Target Combo: A scripted chain of normals into a specific follow-up unique to SF3.; Dash Right: Quickly stepping or hopping toward the right side of the screen (double-tap RIGHT).; Walk Right: Walking horizontally to the right side of the screen.), Passive_Behavior(Idle: Stands still without input, waiting for an interaction.), Strategy(Offense: Focuses on advancing and chaining attacks to keep the opponent on the back foot.)",
    ),
    "offense_jump_in_attack": (
        "Offense",
        "random/clip_0721264/video.mp4",
        "Offense: Closes the distance quickly to apply pressure and initiate close combat.",
        "NPC: Active_Behavior(Jump Right: Jumping in an arc toward the right side of the screen.; Jumping Attack: A punch or kick thrown while airborne (jump-in or air-to-air).; Crouching Kick: A kick or sweep thrown from a crouching posture (includes low sweeps).; Target Combo: A scripted chain of normals into a specific follow-up unique to SF3.; Super Art: A full-screen flashy super move using a stocked super gauge (e.g., Kasumi Suzaku aerial Kunai rain).), Passive_Behavior(N/A), Strategy(Offense: Closes the distance quickly to apply pressure and initiate close combat.)",
    ),
    "defense_crouch_block": (
        "Defense",
        "random/clip_0025278/video.mp4",
        "Defense: Holds ground with blocks and reactive counters, only striking when an opening appears.",
        "NPC: Active_Behavior(Crouch: Lowering the body into a squatting stance to avoid high attacks or prepare low attacks.; Jump Left: Jumping in an arc toward the left side of the screen.), Passive_Behavior(Take Damage: Absorbs a hit and enters hitstun with health loss.; Knockdown: Falls to the ground after a heavy hit or sweep.; Wake Up: Recovery animation while standing back up after a knockdown.; Block Stun: Frozen briefly while blocking an attack — still safe but locked into guard.; Crouching Block: Passively guards against low and mid attacks from a lowered stance.), Strategy(Defense: Holds ground with blocks and reactive counters, only striking when an opening appears.)",
    ),
    "defense_absorb_evade": (
        "Defense",
        "random/clip_0048919/video.mp4",
        "Defense: Absorbs and evades incoming pressure, recovering safely instead of trading hits.",
        "NPC: Active_Behavior(Crouch: Lowering the body into a squatting stance to avoid high attacks or prepare low attacks.), Passive_Behavior(Crouching Block: Passively guards against low and mid attacks from a lowered stance.; Take Damage: Absorbs a hit and enters hitstun with health loss.; Knockback: Pushed away horizontally by the force of a blocked or landed attack.; Knockdown: Falls to the ground after a heavy hit or sweep.; Wake Up: Recovery animation while standing back up after a knockdown.), Strategy(Defense: Absorbs and evades incoming pressure, recovering safely instead of trading hits.)",
    ),
    "control_walk_spacing": (
        "Control",
        "random/clip_0577140/video.mp4",
        "Control: Manages spacing with projectiles and measured pokes to dictate the pace of engagement.",
        "NPC: Active_Behavior(Walk Right: Walking horizontally to the right side of the screen.; Walk Left: Walking horizontally to the left side of the screen.; Crouch: Lowering the body into a squatting stance to avoid high attacks or prepare low attacks.), Passive_Behavior(Idle: Stands still without input, waiting for an interaction.), Strategy(Control: Manages spacing with projectiles and measured pokes to dictate the pace of engagement.)",
    ),
    "control_zoning_kunai": (
        "Control",
        "random/clip_0481531/video.mp4",
        "Control: Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.",
        "NPC: Active_Behavior(Jump Right: Jumping in an arc toward the right side of the screen.; Jump Left: Jumping in an arc toward the left side of the screen.; Kunai Throw: Ibuki's signature throwing-knife projectile, often performed in mid-air at a downward angle.), Passive_Behavior(Evade: Sidesteps or otherwise avoids an incoming attack/projectile without blocking.), Strategy(Control: Uses range and zoning tools to keep the opponent at a preferred distance and force reactions.)",
    ),
}


# Single neutral action preset (no buttons pressed across all 4 segments) so
# inter-job differences are dominated by the prompt.
NEUTRAL_PRESET = {"kind": "segments", "segments": [[], [], [], []]}


def write_report(out_dir: Path) -> None:
    rows = []
    for j in sorted(out_dir.glob("*.json")):
        if j.name in ("_run_args.json", "summary.json"):
            continue
        rows.append(json.loads(j.read_text()))
    if not rows:
        print("[eval_strategy_6] no job outputs found.")
        return
    rows.sort(key=lambda r: (r["extra"]["category"], r["job_id"]))
    lines = [
        "# SF3 — Strategy axis evaluation (6 jobs)",
        "",
        "Each row uses an exact training-set prompt (verbose Active+Passive+Strategy)",
        "with a neutral action preset (no buttons pressed). Active_Behavior of each",
        "prompt is semantically aligned with its Strategy category:",
        "  - Offense: punches / kicks / jump-in attacks",
        "  - Defense: Crouch + blocks (passive-heavy)",
        "  - Control: Walk / Jump / projectile (Kunai)",
        "",
        "`me_right` = NPC-half motion energy. Expect Offense > Control > Defense",
        "if cross_attn carries strategy signal.",
        "",
        "| job_id | category | source_clip | strategy | me_full | me_left | me_right |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        m = r["metrics"]
        e = r["extra"]
        lines.append(
            f"| {r['job_id']} | {e['category']} | {e['source_clip']} | {e['strategy']} | "
            f"{m['me_full']:.3f} | {m['me_left']:.3f} | {m['me_right']:.3f} |"
        )
    lines.append("")
    lines.append("Sanity: me_right should differ across categories.")
    (out_dir / "report.md").write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps({"jobs": rows}, indent=2, ensure_ascii=False))
    print(f"[eval_strategy_6] report.md + summary.json -> {out_dir}")


def main():
    p = argparse.ArgumentParser(description="SF3 strategy-axis eval (slim, 6 jobs)")
    p.add_argument("--full_ckpt", default=None)
    p.add_argument("--lora_ckpt", default=None)
    p.add_argument("--lora_alpha", type=float, default=0.8)
    p.add_argument("--base_model_dir",
                   default="/home/zeqingwang/zeqingwang/models/base_model")
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
    for slug, (cat, src_clip, strategy, full_prompt) in SUB_STRATEGIES.items():
        jobs.append(Job(
            job_id=slug,
            prompt=full_prompt,
            action_spec=NEUTRAL_PRESET,
            input_image_path=args.reference_clip,
            seed=args.seed,
            extra={
                "strategy": strategy,
                "category": cat,
                "source_clip": src_clip,
                "action_preset": "neutral",
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
    print(f"[eval_strategy_6] {len(jobs)} jobs across GPUs {gpu_ids}")
    run_jobs(jobs, cfg, gpu_ids, str(_REACTIVE_GWM))
    write_report(out_dir)


if __name__ == "__main__":
    main()
