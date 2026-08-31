"""Global camera-motion measurement via cv2.phaseCorrelate.

Prints mean per-frame displacement over [start, end):
    MOTION |v|=<mag> dx=<x> dy=<y>  n=<frames>
Image coords: dx>0 = scene shifts right (player moves A/west),
              dy>0 = scene shifts down  (player moves W/north).
Compare a generated clip against the GT clip measured with the SAME tool;
sign convention then cancels out of any ratio test.

Usage: python scripts/measure_motion.py video.mp4 [--start 0] [--end -1]
"""
import argparse

import cv2
import imageio.v3 as iio
import numpy as np


def measure(path, start=0, end=None):
    frames = iio.imread(path)  # [T, H, W, C]
    if end is None or end < 0:
        end = len(frames)
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
         for f in frames[start:end]]
    win = cv2.createHanningWindow(g[0].shape[::-1], cv2.CV_32F)
    shifts = []
    for a, b in zip(g[:-1], g[1:]):
        (sx, sy), _ = cv2.phaseCorrelate(a, b, win)
        shifts.append((sx, sy))
    s = np.array(shifts)
    # median: robust against junk shifts from generation artifacts
    dx, dy = float(np.median(s[:, 0])), float(np.median(s[:, 1]))
    mag = float(np.hypot(s[:, 0], s[:, 1]).mean())
    return mag, dx, dy, len(s)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    a = p.parse_args()
    mag, dx, dy, n = measure(a.video, a.start, a.end)
    print(f"MOTION |v|={mag:.2f} dx={dx:+.2f} dy={dy:+.2f} n={n}")
