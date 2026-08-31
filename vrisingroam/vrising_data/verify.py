"""Alignment verification for cut clips.

1. `overlay`: sample random clips and burn a WASD+Mouse0 HUD into each frame
   from the paired parquet — eyeball that the character moves the way the HUD
   says, on the SAME frame (no lag).

2. `coherence`: the 1 Hz state log gives ground-truth displacement (dx, dz).
   The input log gives an intended move vector u = (D-A, W-S) in screen space.
   Camera->world rotation is unknown but constant (fixed isometric camera), so
   fit a single 2x2 scaled-rotation R over all samples (least squares) and
   report cos(R u, v) statistics, bucketed by hour — a time-dependent drop
   means clock drift between logs and video; a uniformly low value means the
   alignment is broken everywhere.

Usage:
  python -m vrising_data.verify overlay   --session_dir ... --out_root ... [--n 8]
  python -m vrising_data.verify coherence --session_dir ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess

import numpy as np
import pandas as pd

from .actions import ActionTrack, BUTTON_COLS
from .filters import BadIntervals
from .timeline import log_offset_for, parse_vt

import cv2


def overlay_clip(video_path: str, parquet_path: str, out_path: str, fps: int = 20):
    df = pd.read_parquet(parquet_path)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    layout = {"W": (60, h - 90), "A": (20, h - 50), "S": (60, h - 50),
              "D": (100, h - 50), "MOUSE0": (150, h - 50)}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(df):
            break
        row = df.iloc[i]
        for key, (x, y) in layout.items():
            on = row[key] > 0.5
            color = (0, 220, 0) if on else (80, 80, 80)
            cv2.rectangle(frame, (x - 2, y - 28), (x + 36, y + 8), color,
                          -1 if on else 1)
            cv2.putText(frame, key[:2], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"f={i}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        vw.write(frame)
        i += 1
    cap.release()
    vw.release()


def cmd_overlay(args):
    meta = pd.read_csv(os.path.join(args.out_root, "metadata.csv"))
    rows = meta.sample(min(args.n, len(meta)), random_state=args.seed)
    out_dir = os.path.join(args.out_root, "verify")
    os.makedirs(out_dir, exist_ok=True)
    for _, r in rows.iterrows():
        name = os.path.basename(r["video"]).replace(".mp4", "_hud.mp4")
        overlay_clip(os.path.join(args.out_root, r["video"]),
                     os.path.join(args.out_root, r["action"]),
                     os.path.join(out_dir, name))
        print(f"[overlay] {out_dir}/{name}")


def cmd_coherence(args):
    off = log_offset_for(args.session_dir)
    with open(os.path.join(args.session_dir, "状态日志.json"), encoding="utf-8-sig") as f:
        states = json.load(f)
    track = ActionTrack(os.path.join(args.session_dir, "输入日志.json"))
    bad = BadIntervals(os.path.join(args.session_dir, "状态日志.json"))

    samples = []  # (t, u=(D-A, W-S) mean over [t_prev,t], v=(dx,dz))
    prev = None
    for s in states:
        if s.get("实体类型") != "玩家" or s.get("视频时间") is None:
            continue
        t = parse_vt(s["视频时间"]) - off
        p = s.get("位置") or {}
        pos = np.array([p.get("x", 0.0), p.get("z", 0.0)])
        if prev is not None:
            t0, p0 = prev
            dt = t - t0
            if 0.5 < dt < 2.0 and not bad.overlaps(t0, t):
                act = track.rasterize(t0, max(int(dt * 20), 1), 20.0)
                u = np.array([act[:, 3].mean() - act[:, 1].mean(),   # D - A
                              act[:, 0].mean() - act[:, 2].mean()])  # W - S
                samples.append((t0, u, (pos - p0) / dt))
        prev = (t, pos)

    U = np.array([u for _, u, _ in samples])
    V = np.array([v for _, _, v in samples])
    ts = np.array([t for t, _, _ in samples])
    mask = (np.linalg.norm(U, axis=1) > 0.3) & (np.linalg.norm(V, axis=1) > 0.5)
    U, V, ts = U[mask], V[mask], ts[mask]
    print(f"[coherence] usable samples: {len(U)}")

    # Least-squares scaled rotation R (2x2: [[a,-b],[b,a]]) s.t. R@u ~ v.
    su = (U * U).sum()
    a = (U[:, 0] * V[:, 0] + U[:, 1] * V[:, 1]).sum() / su
    b = (U[:, 0] * V[:, 1] - U[:, 1] * V[:, 0]).sum() / su
    R = np.array([[a, -b], [b, a]])
    P = U @ R.T
    cos = (P * V).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(V, axis=1) + 1e-9)
    ang = np.degrees(np.arctan2(b, a))
    print(f"[coherence] fitted camera rotation={ang:.1f} deg, "
          f"scale={np.hypot(a, b):.2f} m/s per unit key")
    print(f"[coherence] cosine: mean={cos.mean():.3f} median={np.median(cos):.3f} "
          f"p10={np.percentile(cos, 10):.3f}  (>0.8 mean = aligned)")
    for h in range(0, int(ts.max() // 3600) + 1):
        m = (ts >= h * 3600) & (ts < (h + 1) * 3600)
        if m.sum() > 50:
            print(f"  hour {h:02d}: n={m.sum():5d} mean_cos={cos[m].mean():.3f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    po = sub.add_parser("overlay")
    po.add_argument("--session_dir", required=True)
    po.add_argument("--out_root", required=True)
    po.add_argument("--n", type=int, default=8)
    po.add_argument("--seed", type=int, default=0)
    pc = sub.add_parser("coherence")
    pc.add_argument("--session_dir", required=True)
    args = p.parse_args()
    {"overlay": cmd_overlay, "coherence": cmd_coherence}[args.cmd](args)


if __name__ == "__main__":
    main()
