"""Virtual timeline over the chunked session video.

Ground truth is `frame_audit.json`: per chunk we know `source_start_sec`
(position of its frame 0 on the original OBS recording), `skip_leading_frames`
(frames duplicated from the previous chunk that must be skipped when
reconstructing the source), and `start_is_measured` (False for chunk_000-004,
whose offsets were estimated after the fact — excluded by default because a
few-frame misalignment poisons action labels).

All log "视频时间" strings (HH:MM:SS:FF @ 60fps) live on the same source-video
clock, so a clip cut at source second T can look up actions directly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

SOURCE_FPS = 60.0


def parse_vt(s: str) -> float:
    """'HH:MM:SS:FF' (FF = frame index @60fps) -> source seconds."""
    h, m, sec, f = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(f) / SOURCE_FPS


# Per-session log-clock offset: 视频时间 stamps minus actual video frames,
# measured by cross-correlating scene motion energy against the input log /
# state-log player speed (scratchpad chunk_align_check{,2}.py).
#   20260731: +1.725 s (OBS started ~1.72 s after the mod's reference instant;
#             stable across the whole 33 h session, measured 2026-08-04).
#   20260730: 0 (newer logger build; measured 2026-08-05, peak at +50 ms).
# Every new session MUST be measured before processing — do not assume.
SESSION_LOG_OFFSETS = {
    "20260731_213546_491": 103.5 / SOURCE_FPS,  # 1.725 s
    "20260730_182621_420": 0.0,
    "20260806_140321_078": 0.0,  # eval-only session; measured 2026-08-06, peak +100 ms (render latency)
}


def log_offset_for(path_or_id) -> float:
    """Resolve the measured log-clock offset from any path containing the
    session id. Raises for unmeasured sessions rather than guessing."""
    s = str(path_or_id)
    for sid, off in SESSION_LOG_OFFSETS.items():
        if sid in s:
            return off
    raise KeyError(
        f"no measured log-clock offset for session of {s!r}; measure it "
        f"with chunk_align_check2 and add it to SESSION_LOG_OFFSETS"
    )


@dataclass(frozen=True)
class ChunkSpan:
    name: str                 # e.g. "chunk_005.mp4"
    source_start_sec: float   # source time of local frame 0
    last_frame_sec: float     # source time of the final frame
    skip_leading_frames: int  # duplicated frames at the head
    start_is_measured: bool

    @property
    def usable_start_sec(self) -> float:
        """First source second not duplicated from the previous chunk."""
        return self.source_start_sec + self.skip_leading_frames / SOURCE_FPS

    def to_local_sec(self, source_sec: float) -> float:
        return source_sec - self.source_start_sec


class Timeline:
    def __init__(self, session_dir: str, include_unmeasured: bool = False):
        with open(os.path.join(session_dir, "frame_audit.json"), encoding="utf-8-sig") as f:
            audit = json.load(f)
        self.fps = audit["video"]["fps"]
        self.duration_sec = audit["video"]["duration_sec"]
        self.chunks: list[ChunkSpan] = []
        for name, c in sorted(audit["chunks"].items()):
            span = ChunkSpan(
                name=name,
                source_start_sec=c["source_start_sec"],
                last_frame_sec=c["last_frame_sec"],
                skip_leading_frames=c["skip_leading_frames"],
                start_is_measured=c["start_is_measured"],
            )
            if span.start_is_measured or include_unmeasured:
                self.chunks.append(span)

    def get(self, name: str) -> ChunkSpan:
        for c in self.chunks:
            if c.name == name or c.name == name + ".mp4":
                return c
        raise KeyError(f"{name} not in timeline (excluded or unknown)")
