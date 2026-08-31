"""Bin danze's per-frame ABot action npy -> per-LTX-latent-frame sentence ids.

Input:  clips/<prefix>/<id>_w000.npy   [130, 17] float32 (frame-level, unscaled)
        channel order: W A S D Q E I J K L Space | d_pitch d_yaw d_roll | d_x d_y d_z
Output: same path with suffix .ltxids.npy  [16, 2] int8  (move_id, view_id)

LTX-2.3 latent frames for 121 pixel frames: frame 0 alone (causal VAE), then
15 groups of 8 -> latent_t = 16. Keys binned with amax over each group
(pressed anywhere in the span counts), then mapped to the 9-combo vocabulary
per segment (movement from W/A/S/D with W-S and A-D cancellation, view from
I/J/K/L likewise).
"""
import argparse
import glob
import os

import numpy as np

KEY = {k: i for i, k in enumerate(["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"])}
MOVE_KEYS = ["", "W", "S", "A", "D", "WA", "WD", "SA", "SD"]
VIEW_KEYS = ["", "J", "L", "I", "K", "JI", "JK", "LI", "LK"]
MOVE_ID = {k: i for i, k in enumerate(MOVE_KEYS)}
VIEW_ID = {k: i for i, k in enumerate(VIEW_KEYS)}

GROUPS = [(0, 1)] + [(1 + 8 * g, 9 + 8 * g) for g in range(15)]   # 16 spans over 121 frames


def combo_move(w, a, s, d):
    fwd = "W" if (w and not s) else ("S" if (s and not w) else "")
    lat = "A" if (a and not d) else ("D" if (d and not a) else "")
    return (fwd + lat) if (fwd + lat) in MOVE_ID else fwd or lat or ""


def combo_view(i, j, k, l):
    yaw = "J" if (j and not l) else ("L" if (l and not j) else "")
    pit = "I" if (i and not k) else ("K" if (k and not i) else "")
    return (yaw + pit) if (yaw + pit) in VIEW_ID else yaw or pit or ""


def bin_one(npy_path):
    a = np.load(npy_path)[:121]                     # [121, 17]
    ids = np.zeros((16, 2), dtype=np.int8)
    for g, (s, e) in enumerate(GROUPS):
        span = a[s:e]
        k = (span.max(axis=0) > 0.5)
        mv = combo_move(k[KEY["W"]], k[KEY["A"]], k[KEY["S"]], k[KEY["D"]])
        vw = combo_view(k[KEY["I"]], k[KEY["J"]], k[KEY["K"]], k[KEY["L"]])
        ids[g, 0] = MOVE_ID[mv]
        ids[g, 1] = VIEW_ID[vw]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_root", default="/nfs/danze/data/abot/clips")
    ap.add_argument("--out_root", default="/data/yuzhewu/ltxwm/data/actions")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.clips_root, "*", "*_w000.npy")))
    if args.limit:
        files = files[: args.limit]
    n_done = 0
    stats = np.zeros((2, 9), dtype=np.int64)
    for f in files:
        rel = os.path.relpath(f, args.clips_root)
        out = os.path.join(args.out_root, rel.replace(".npy", ".ltxids.npy"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ids = bin_one(f)
        np.save(out, ids)
        for c in range(2):
            for v in ids[:, c]:
                stats[c, v] += 1
        n_done += 1
    print(f"binned {n_done} clips -> {args.out_root}")
    print("move id histogram:", dict(zip(MOVE_KEYS, stats[0].tolist())))
    print("view id histogram:", dict(zip(VIEW_KEYS, stats[1].tolist())))


if __name__ == "__main__":
    main()
