"""Rasterize the press/release input log into per-frame action rows.

Input log entries: {"视频时间": "HH:MM:SS:FF", "按键": "W|A|S|D|Mouse0",
"状态": "按下|松开"} on the source-video clock (60 fps).

Output schema (one row per target-fps frame) — column order is the dim order
the model sees, so it is frozen here and mirrored by the `vrising` GameProfile:

    W, A, S, D, MOUSE0            binary key state (OR over the covered window)
    CAM_X, CAM_Y                  reserved: camera/mouse deltas (0 for sessions
                                  whose 鼠标日志 was lost)
    CAM_ACTIVE                    1 if CAM_X/CAM_Y carry real data, else 0

A key counts as held for target frame j (window [t0 + j/fps, t0 + (j+1)/fps))
if it is down at any of the 60 fps source-frame instants inside the window, so
a 0.02 s Mouse0 click is never lost to downsampling.
"""
from __future__ import annotations

import json

import numpy as np

from .timeline import SOURCE_FPS, log_offset_for, parse_vt

BUTTON_COLS = ("W", "A", "S", "D", "MOUSE0")
CAM_COLS = ("CAM_X", "CAM_Y", "CAM_ACTIVE")
ALL_COLS = BUTTON_COLS + CAM_COLS

_KEY_TO_COL = {"W": 0, "A": 1, "S": 2, "D": 3, "Mouse0": 4}


class ActionTrack:
    """Per-key sorted [down_sec, up_sec) intervals on the source clock."""

    def __init__(self, input_log_path: str):
        off = log_offset_for(input_log_path)
        with open(input_log_path, encoding="utf-8-sig") as f:
            events = json.load(f)
        open_at: dict[str, float] = {}
        intervals: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(BUTTON_COLS))}
        for e in events:
            key = e["按键"]
            col = _KEY_TO_COL.get(key)
            if col is None or e["视频时间"] is None:
                continue
            t = parse_vt(e["视频时间"]) - off
            if e["状态"] == "按下":
                open_at[key] = t
            elif key in open_at:
                intervals[col].append((open_at.pop(key), t))
        # A press with no matching release (recording cut) holds to +inf.
        for key, t in open_at.items():
            intervals[_KEY_TO_COL[key]].append((t, float("inf")))
        self._starts = {}
        self._ends = {}
        for col, ivs in intervals.items():
            ivs.sort()
            self._starts[col] = np.array([a for a, _ in ivs])
            self._ends[col] = np.array([b for _, b in ivs])

    def _down_at(self, col: int, times: np.ndarray) -> np.ndarray:
        """Vectorized: is key `col` held at each source-time instant?"""
        idx = np.searchsorted(self._starts[col], times, side="right") - 1
        ok = idx >= 0
        down = np.zeros(len(times), dtype=bool)
        down[ok] = times[ok] < self._ends[col][idx[ok]]
        return down

    def rasterize(self, t0: float, num_frames: int, fps: float) -> np.ndarray:
        """[num_frames, len(ALL_COLS)] float32; OR over the source frames each
        target frame covers."""
        sub = int(round(SOURCE_FPS / fps))  # 3 for 60->20
        # instants: frame j sample s -> t0 + (j*sub + s)/60
        inst = t0 + np.arange(num_frames * sub) / SOURCE_FPS
        out = np.zeros((num_frames, len(ALL_COLS)), dtype=np.float32)
        for col in range(len(BUTTON_COLS)):
            down = self._down_at(col, inst).reshape(num_frames, sub)
            out[:, col] = down.any(axis=1)
        return out
