"""Sample 6 clips from metadata_wo_pure_v5cat_10k.csv for strategy listening eval.

Two rows per category (Offense / Defense / Control), preferring different
strategy substrings within a category so the same prompt slot isn't sampled
twice. Reproducible via seed.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


CATEGORIES = ["Offense", "Defense", "Control"]
PER_CATEGORY = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    rng = random.Random(args.seed)

    picked = []
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        if sub.empty:
            raise SystemExit(f"no rows for category {cat}")
        # Group by strategy substring; sample one row per distinct strategy first.
        by_strategy = {s: g.index.tolist() for s, g in sub.groupby("strategy")}
        strategies = list(by_strategy.keys())
        rng.shuffle(strategies)
        chosen_idxs: list[int] = []
        for s in strategies:
            if len(chosen_idxs) >= PER_CATEGORY:
                break
            chosen_idxs.append(rng.choice(by_strategy[s]))
        # If a category has fewer distinct strategies than PER_CATEGORY, top up
        # from the same category.
        while len(chosen_idxs) < PER_CATEGORY:
            chosen_idxs.append(rng.choice(sub.index.tolist()))
        for idx in chosen_idxs:
            picked.append(df.loc[idx])

    out = pd.DataFrame(picked).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"[sampler] wrote {len(out)} rows to {args.out_csv}")
    for _, r in out.iterrows():
        print(f"  [{r['category']:>7}] {r['video']}  ::  {r['strategy'][:80]}")


if __name__ == "__main__":
    main()
