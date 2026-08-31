"""Lightweight region-based metrics for ReactiveGWM evaluation.

Both SF2 and SF3 follow the standard fighting-game framing — Player occupies
the LEFT half of the screen, NPC the RIGHT half. So:
  - Action axis:    changes in player input should move me_left between conditions.
  - Strategy axis:  changes in NPC prompt should move me_right between conditions.

The midline split assumes a centered camera, which both SF2 (Genesis) and SF3
(CPS3 6-panel) use. Render resolution can vary (SF2 480x608 vs SF3 480x832);
the splitting is on width and stays clean either way.
"""
from __future__ import annotations

import numpy as np


def _region_slice(region: str, W: int) -> slice:
    if region == "left":  return slice(0, W // 2)
    if region == "right": return slice(W // 2, W)
    if region == "full":  return slice(0, W)
    raise ValueError(region)


def region_motion_energy(frames: np.ndarray, region: str) -> float:
    """[T, H, W, 3] uint8 -> mean frame-to-frame abs diff inside region."""
    W = frames.shape[2]
    s = _region_slice(region, W)
    f = frames[:, :, s, :].astype(np.float64)
    return float(np.abs(f[1:] - f[:-1]).mean())


def region_mse(a: np.ndarray, b: np.ndarray, region: str) -> float:
    """Two [T, H, W, 3] uint8 arrays -> mean pixel MSE inside region."""
    W = a.shape[2]
    s = _region_slice(region, W)
    T = min(a.shape[0], b.shape[0])
    a2 = a[:T, :, s, :].astype(np.float64)
    b2 = b[:T, :, s, :].astype(np.float64)
    return float(((a2 - b2) ** 2).mean())


def all_regions_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "mse_full":  region_mse(a, b, "full"),
        "mse_left":  region_mse(a, b, "left"),
        "mse_right": region_mse(a, b, "right"),
        "me_a_full":  region_motion_energy(a, "full"),
        "me_a_left":  region_motion_energy(a, "left"),
        "me_a_right": region_motion_energy(a, "right"),
        "me_b_full":  region_motion_energy(b, "full"),
        "me_b_left":  region_motion_energy(b, "left"),
        "me_b_right": region_motion_energy(b, "right"),
    }


def frames_from_pil_list(pil_frames) -> np.ndarray:
    """list[PIL.Image] -> uint8 ndarray [T, H, W, 3]."""
    return np.stack([np.array(f) for f in pil_frames]).astype(np.uint8)
