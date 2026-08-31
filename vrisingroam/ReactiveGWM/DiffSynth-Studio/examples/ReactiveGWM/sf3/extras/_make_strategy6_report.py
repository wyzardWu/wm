"""Generate report.md for the strategy6 eval, grouped by category."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def short_strategy(s: str) -> str:
    return s.split(":", 1)[1].strip().rstrip(".") if ":" in s else s


def short_active(prompt: str) -> str:
    m = re.search(r"Active_Behavior\(([^)]*)\)", prompt)
    if not m:
        return ""
    items = [x.split(":", 1)[0].strip() for x in m.group(1).split(";") if x.strip()]
    return ", ".join(items)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    samples = pd.read_csv(out_dir / "samples.csv")
    rows = []
    for _, r in samples.iterrows():
        clip_dir = Path(r["video"]).parent.as_posix().replace("/", "_")
        meta_path = out_dir / f"{clip_dir}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        rows.append({
            "category": r["category"],
            "strategy": r["strategy"],
            "active": short_active(r["prompt"]),
            "job_id": clip_dir,
            "gen": f"{clip_dir}.mp4",
            "orig": f"{clip_dir}_original.mp4",
            "ok": meta_path.exists() and (out_dir / f"{clip_dir}.mp4").exists(),
        })

    cat_order = ["Offense", "Defense", "Control"]
    rows.sort(key=lambda x: (cat_order.index(x["category"]) if x["category"] in cat_order else 99,
                             x["job_id"]))

    cn = {"Offense": "进攻 (Offense)", "Defense": "防守 (Defense)", "Control": "控场 (Control)"}
    lines = [
        "# SF3 strategy listening eval — step-26000 / cfg=5.0",
        "",
        f"- Checkpoint: `p1_joint_480x832_5s_fixedprompt_coldstart_freeze_xattn/step-26000.safetensors`",
        f"- Sample: 6 clips drawn from `metadata_wo_pure_v5cat_10k.csv` (2 per category, distinct strategy substrings).",
        f"- Inference: 480x832, 101 frames, 30 steps, cfg_scale=5.0, action_cfg_scale=1.0, action_hold_window=10, seed=0.",
        f"- Each job uses the per-clip CSV prompt (containing `Strategy(...)`), the clip's first frame as input image, and the clip's actual button parquet as action conditioning.",
        f"- For each job we keep the generated mp4 and a symlink to the original ground-truth clip for side-by-side review.",
        "",
        "## How to use this report",
        "",
        "Eyeball test: open `<job_id>.mp4` (generated) next to `<job_id>_original.mp4`. The NPC (right-side fighter) should *match the strategy declared in the prompt* — Offense ⇒ pressing forward / striking; Defense ⇒ blocking / retreating / minimal aggression; Control ⇒ measured spacing, neither rushing nor turtling.",
        "",
        "Cross-category check: same active-behavior list with different `Strategy(...)` should yield visibly different NPC tendencies. If the NPC behaves the same regardless of the strategy slot, prompt CFG isn't reaching cross-attention well at this step.",
        "",
    ]

    for cat in cat_order:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        lines.append(f"## {cn.get(cat, cat)}")
        lines.append("")
        for r in cat_rows:
            status = "" if r["ok"] else "  ⚠ MISSING OUTPUT"
            lines.append(f"### `{r['job_id']}`{status}")
            lines.append("")
            lines.append(f"- **Strategy**: {short_strategy(r['strategy'])}")
            lines.append(f"- **Active behaviors in prompt**: {r['active']}")
            lines.append(f"- Generated: [`{r['gen']}`]({r['gen']})")
            lines.append(f"- Original:  [`{r['orig']}`]({r['orig']})")
            lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `samples.csv` — the 6 sampled rows (full prompt text).")
    lines.append("- `_run_args.json` — top-level launcher args.")
    lines.append("- `_shards/row_NN.csv` — per-job 1-row CSV (input to each python subprocess).")
    lines.append("- `_logs/row_NN_gpuM.log` — per-job inference log.")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"[report] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
