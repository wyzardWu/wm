"""Cut a session's video chunks into training clips + per-frame action parquets.

One sequential decode per chunk (fps-downsample + resize in ffmpeg), then the
20 fps frame stream is sliced into fixed-length windows. Windows overlapping a
bad interval (teleport / death / log gap) or containing no movement input are
dropped. Surviving windows are re-encoded as short mp4s by a worker pool and
paired with a [num_frames, 8] action parquet on the exact same time grid.

Usage:
  python -m vrising_data.cut_clips \
      --session_dir data/raw/20260731_213546_491 \
      --out_root    data/processed/20260731 \
      --chunks      chunk_005            # omit = all measured chunks
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .actions import ALL_COLS, BUTTON_COLS, ActionTrack
from .filters import BadIntervals
from .timeline import SOURCE_FPS, Timeline

FFMPEG = "ffmpeg"


def decode_stream(video_path: str, width: int, height: int, fps: int):
    """Yield (decoded_frame_index, rgb24 bytes) at `fps`, resized/cropped."""
    # Native center crop from the 1080p source (decided 2026-08-05): no
    # downscale — the character stays large and the HUD falls outside the
    # crop. Requires source frames >= width x height.
    vf = f"fps={fps},crop={width}:{height}"
    proc = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", video_path,
         "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=10 ** 8,
    )
    frame_bytes = width * height * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield i, buf
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def encode_clip(frames: list[bytes], out_path: str, width: int, height: int,
                fps: int, crf: int = 18) -> None:
    proc = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-g", "20", out_path],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        proc.stdin.write(f)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg encode failed: {out_path}")


def process_chunk(chunk, session_dir, out_root, actions, bad,
                  width, height, fps, num_frames, stride, min_move_frames,
                  encoder_pool, meta_rows, session_id):
    video_path = os.path.join(session_dir, "video", chunk.name)
    chunk_stem = chunk.name.removesuffix(".mp4")
    clip_dir = os.path.join(out_root, "clips", chunk_stem)
    act_dir = os.path.join(out_root, "actions", chunk_stem)
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(act_dir, exist_ok=True)

    sub = int(round(SOURCE_FPS / fps))
    first_j = -(-chunk.skip_leading_frames // sub)  # skip duplicated head
    window: list[bytes] = []
    win_start_j = first_j
    futures, kept, dropped_bad, dropped_idle = [], 0, 0, 0

    for j, frame in decode_stream(video_path, width, height, fps):
        if j < first_j:
            continue
        if j < win_start_j:  # window advanced past this frame (stride > len)
            continue
        window.append(frame)
        if len(window) < num_frames:
            continue

        t0 = chunk.source_start_sec + win_start_j / fps
        t1 = t0 + num_frames / fps
        act = actions.rasterize(t0, num_frames, fps)
        if bad.overlaps(t0, t1):
            dropped_bad += 1
        elif act[:, :4].sum() < min_move_frames:
            dropped_idle += 1
        else:
            g0 = int(round(t0 * SOURCE_FPS))
            name = f"{session_id}_{chunk_stem}_{g0:08d}"
            clip_rel = f"clips/{chunk_stem}/{name}.mp4"
            act_rel = f"actions/{chunk_stem}/{name}.parquet"
            pd.DataFrame(act, columns=list(ALL_COLS)).to_parquet(
                os.path.join(out_root, act_rel))
            futures.append(encoder_pool.submit(
                encode_clip, list(window), os.path.join(out_root, clip_rel),
                width, height, fps))
            meta_rows.append({
                "video": clip_rel, "action": act_rel, "prompt": "",
                "session": session_id, "chunk": chunk_stem,
                "source_start_sec": round(t0, 6),
                "move_frames": int(act[:, :4].any(axis=1).sum()),
                "mouse0_frames": int(act[:, 4].sum()),
            })
            kept += 1

        # advance: window currently holds [win_start_j, win_start_j+num_frames)
        win_start_j += stride
        window = window[stride:] if stride < num_frames else []

    for f in futures:
        f.result()
    return kept, dropped_bad, dropped_idle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--chunks", nargs="*", default=None,
                   help="chunk names (default: every measured chunk present on disk)")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--num_frames", type=int, default=101)
    p.add_argument("--stride", type=int, default=101,
                   help="window stride in target-fps frames (< num_frames = overlap)")
    p.add_argument("--min_move_frames", type=int, default=10,
                   help="min summed WASD-active frames to keep a clip")
    p.add_argument("--encoders", type=int, default=16)
    args = p.parse_args()

    session_id = os.path.basename(os.path.normpath(args.session_dir))
    tl = Timeline(args.session_dir)
    actions = ActionTrack(os.path.join(args.session_dir, "输入日志.json"))
    bad = BadIntervals(os.path.join(args.session_dir, "状态日志.json"))
    print(f"[cut] bad intervals: {len(bad.intervals)} covering "
          f"{bad.total_bad_sec():.0f}s of {tl.duration_sec:.0f}s")

    if args.chunks:
        chunks = [tl.get(c) for c in args.chunks]
    else:
        chunks = [c for c in tl.chunks if os.path.exists(
            os.path.join(args.session_dir, "video", c.name))]
    os.makedirs(args.out_root, exist_ok=True)

    meta_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.encoders) as pool:
        for c in chunks:
            kept, dbad, didle = process_chunk(
                c, args.session_dir, args.out_root, actions, bad,
                args.width, args.height, args.fps, args.num_frames,
                args.stride, args.min_move_frames, pool, meta_rows, session_id)
            print(f"[cut] {c.name}: kept={kept} bad={dbad} idle={didle}")

    meta_path = os.path.join(args.out_root, "metadata.csv")
    exists = os.path.exists(meta_path)
    old = pd.read_csv(meta_path).to_dict("records") if exists else []
    seen = {r["video"] for r in old}
    merged = old + [r for r in meta_rows if r["video"] not in seen]
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    print(f"[cut] metadata.csv: {len(merged)} rows ({len(meta_rows)} new)")


if __name__ == "__main__":
    main()
