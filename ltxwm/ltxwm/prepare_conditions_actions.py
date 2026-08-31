"""Inject per-clip action_ids into the trainer's precomputed conditions .pt files.

Run AFTER the official process_dataset.py produced conditions/. Matches each
condition file to its clip via the dataset manifest order (both are produced
from the same file list), loads our .ltxids.npy, and adds
  cond["action_ids"] : int8 [16, 2]
Idempotent: skips files that already carry action_ids.

Usage: python prepare_conditions_actions.py --data_root <preprocessed_root> \
           --actions_root /data/yuzhewu/ltxwm/data/actions --manifest <jsonl>
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--actions_root", default="/data/yuzhewu/ltxwm/data/actions")
    ap.add_argument("--manifest", required=True,
                    help="jsonl in the SAME order process_dataset consumed (video field)")
    args = ap.parse_args()

    cond_dir = os.path.join(args.data_root, "conditions")
    rows = [json.loads(line) for line in open(args.manifest)]

    n_add = 0
    for row in rows:
        # condition files mirror the ABSOLUTE source video path under conditions/
        p = os.path.join(cond_dir, row["video"].lstrip("/")).replace(".mp4", ".pt")
        assert os.path.exists(p), f"missing condition file {p}"
        d = torch.load(p, map_location="cpu", weights_only=True)
        if "action_ids" in d:
            continue
        rel = os.path.relpath(row["video"], "/nfs/danze/data/abot/clips").replace(".mp4", ".ltxids.npy")
        ids = np.load(os.path.join(args.actions_root, rel))
        d["action_ids"] = torch.from_numpy(ids)      # int8 [16, 2]
        torch.save(d, p)
        n_add += 1
    print(f"injected action_ids into {n_add}/{len(rows)} condition files")


if __name__ == "__main__":
    main()
