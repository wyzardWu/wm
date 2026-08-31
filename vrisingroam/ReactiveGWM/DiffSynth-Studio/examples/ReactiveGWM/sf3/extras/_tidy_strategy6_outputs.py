"""Rename strategy6 output mp4s by strategy slug and consolidate metadata.

After running `run_strategy6_step26000.sh` the output dir has per-clip
filenames like `random_clip_<id>.mp4` plus original symlinks plus per-clip
jsons. This script:
  - renames each generated mp4 to `<category>_<slug>.mp4` (slug taken from the
    v5cat strategy substring, matching eval_strategy.py SUB_STRATEGIES keys)
  - deletes the `_original.mp4` symlinks
  - merges all per-clip jsons into a single `meta.json` and removes the
    individual ones
  - rewrites `report.md` with the new names
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# Strategy substring -> slug. Keys must match the *exact* strategy text in
# metadata_wo_pure_v5cat_10k.csv (verified against the file).
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


def short_active(prompt: str) -> list[str]:
    m = re.search(r"Active_Behavior\(([^)]*)\)", prompt)
    if not m:
        return []
    return [x.split(":", 1)[0].strip() for x in m.group(1).split(";") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    samples = pd.read_csv(out_dir / "samples.csv")
    entries = []

    for _, r in samples.iterrows():
        old_id = Path(r["video"]).parent.as_posix().replace("/", "_")
        old_mp4 = out_dir / f"{old_id}.mp4"
        old_orig = out_dir / f"{old_id}_original.mp4"
        old_json = out_dir / f"{old_id}.json"

        slug = STRATEGY_SLUGS.get(r["strategy"].strip())
        if slug is None:
            raise SystemExit(f"unknown strategy: {r['strategy']!r}")

        new_mp4 = out_dir / f"{slug}.mp4"

        meta = json.loads(old_json.read_text()) if old_json.exists() else {}

        if old_mp4.exists():
            old_mp4.rename(new_mp4)
        if old_orig.exists() or old_orig.is_symlink():
            old_orig.unlink()
        if old_json.exists():
            old_json.unlink()

        entries.append({
            "slug": slug,
            "category": r["category"],
            "strategy": r["strategy"],
            "active_behaviors": short_active(r["prompt"]),
            "video": f"{slug}.mp4",
            "source_clip": r["video"],
            "source_action": r["action"],
            "prompt": r["prompt"],
            "cfg_scale": meta.get("cfg_scale"),
            "action_cfg_scale": meta.get("action_cfg_scale"),
            "num_frames": meta.get("num_frames"),
            "num_inference_steps": meta.get("num_inference_steps"),
            "height": meta.get("height"),
            "width": meta.get("width"),
            "seed": meta.get("seed"),
            "ckpt": meta.get("ckpt"),
        })

    cat_order = ["Offense", "Defense", "Control"]
    entries.sort(key=lambda e: (cat_order.index(e["category"]) if e["category"] in cat_order else 99,
                                e["slug"]))

    (out_dir / "meta.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    cn = {"Offense": "进攻 (Offense)", "Defense": "防守 (Defense)", "Control": "控场 (Control)"}
    lines = [
        "# SF3 strategy listening eval — step-26000 / cfg=5.0",
        "",
        "Each video tests one strategy substring (NPC = right-side fighter). Filename = `<category>_<slug>.mp4`.",
        "",
        f"- Checkpoint: `p1_joint_480x832_5s_fixedprompt_coldstart_freeze_xattn/step-26000.safetensors`",
        f"- 6 clips drawn from `metadata_wo_pure_v5cat_10k.csv` (2 per category, distinct strategy substrings).",
        f"- Inference: 480x832, 101 frames, 30 steps, cfg_scale=5.0, action_cfg_scale=1.0, action_hold_window=10, seed=0.",
        "",
        "Per video the prompt is the per-clip CSV prompt (containing `Strategy(...)`); the input image is the source clip's first frame; the action stream is the source clip's actual button parquet.",
        "",
    ]
    for cat in cat_order:
        cat_rows = [e for e in entries if e["category"] == cat]
        if not cat_rows:
            continue
        lines.append(f"## {cn.get(cat, cat)}")
        lines.append("")
        for e in cat_rows:
            strat = e["strategy"].split(":", 1)[1].strip().rstrip(".") if ":" in e["strategy"] else e["strategy"]
            lines.append(f"### `{e['slug']}` → [{e['video']}]({e['video']})")
            lines.append(f"- {strat}")
            lines.append(f"- Active behaviors: {', '.join(e['active_behaviors'])}")
            lines.append(f"- Source clip: `{e['source_clip']}`")
            lines.append("")

    lines += [
        "## Files",
        "",
        "- `<category>_<slug>.mp4` — generated videos.",
        "- `meta.json` — combined metadata for all 6 jobs.",
        "- `samples.csv` — original sampled rows (kept for traceability).",
        "- `_run_args.json` — top-level launcher args.",
        "- `_shards/`, `_logs/` — per-job inference shards and logs.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"[tidy] renamed {len(entries)} videos, wrote meta.json + report.md")


if __name__ == "__main__":
    main()
