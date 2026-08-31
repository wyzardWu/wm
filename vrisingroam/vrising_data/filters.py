"""Bad-interval detection from the 1 Hz state log and the event log.

A clip window is rejected if it overlaps any interval where the recording is
not clean roaming footage:

  - teleport / map load: player displacement between consecutive 1 Hz state
    samples exceeds `teleport_speed` (m/s), or the state log itself has a gap
    longer than `state_gap_sec` (loading screens stop the logger).
  - death: 生死状态 != 存活 or 血量 <= 0 (death warps the camera to the coffin).

Each hit is padded by `margin_sec` on both sides, then intervals are merged.
"""
from __future__ import annotations

import json

from .timeline import log_offset_for, parse_vt


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals.sort()
    out = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


class BadIntervals:
    def __init__(
        self,
        state_log_path: str,
        teleport_speed: float = 15.0,
        state_gap_sec: float = 3.0,
        margin_sec: float = 2.0,
    ):
        off = log_offset_for(state_log_path)
        with open(state_log_path, encoding="utf-8-sig") as f:
            states = json.load(f)
        bad: list[tuple[float, float]] = []
        prev_t, prev_p = None, None
        for s in states:
            if s.get("实体类型") != "玩家" or s.get("视频时间") is None:
                continue
            t = parse_vt(s["视频时间"]) - off
            p = s.get("位置") or {}
            pos = (p.get("x", 0.0), p.get("z", 0.0))  # ground plane; y is height
            if s.get("生死状态") != "存活" or (s.get("血量") or 1) <= 0:
                bad.append((t - margin_sec, t + margin_sec))
            if prev_t is not None:
                dt = t - prev_t
                if dt > state_gap_sec:
                    bad.append((prev_t - margin_sec, t + margin_sec))
                elif dt > 0:
                    d = ((pos[0] - prev_p[0]) ** 2 + (pos[1] - prev_p[1]) ** 2) ** 0.5
                    if d / dt > teleport_speed:
                        bad.append((prev_t - margin_sec, t + margin_sec))
            prev_t, prev_p = t, pos
        self.intervals = _merge(bad)

    def overlaps(self, a: float, b: float) -> bool:
        for x, y in self.intervals:
            if x < b and a < y:
                return True
            if x >= b:
                break
        return False

    def total_bad_sec(self) -> float:
        return sum(b - a for a, b in self.intervals)
