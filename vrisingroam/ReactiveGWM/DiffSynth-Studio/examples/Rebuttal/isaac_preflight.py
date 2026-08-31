#!/usr/bin/env python3
"""CPU-only full-dataset audit for the locked Isaac training contract."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.Rebuttal.checkpoint_io import atomic_write_json
from examples.Rebuttal.isaac_profile import read_isaac_action
from examples.Rebuttal.preflight import validate_metadata
from examples.Rebuttal.variants import ISAAC_DATA_ROOT, ISAAC_METADATA, VARIANTS


DEFAULT_REPORT = HERE / "generated/isaac_data_preflight.json"
DEFAULT_SMOKE_METADATA = HERE / "generated/metadata_isaac_v1_smoke16.csv"


def _resolve_under(root: Path, relative: str, kind: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{kind} path escapes dataset root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{kind} file is missing: {path}")
    return path


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream, got {len(streams)}: {path}")
    stream = streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/", maxsplit=1)
    fps = float(numerator) / float(denominator)
    frames_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    frames = int(frames_raw)
    actual = {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": frames,
        "fps": fps,
    }
    expected = {"width": 832, "height": 480, "frames": 101}
    mismatches = {
        key: {"actual": actual[key], "expected": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if abs(fps - 20.0) > 1e-6:
        mismatches["fps"] = {"actual": fps, "expected": 20.0}
    if mismatches:
        raise ValueError(f"Isaac video contract mismatch {mismatches}: {path}")
    return actual


def _audit_one(root: Path, video: str, action: str) -> dict[str, Any]:
    video_path = _resolve_under(root, video, "video")
    action_path = _resolve_under(root, action, "action")
    action_tensor = read_isaac_action(action_path)
    video_info = _probe_video(video_path)
    return {
        "video": video,
        "action": action,
        "action_shape": list(action_tensor.shape),
        "video_info": video_info,
    }


def run_preflight(
    *,
    data_root: Path,
    metadata_path: Path,
    report_path: Path,
    smoke_metadata_path: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.time()
    metadata = validate_metadata(VARIANTS["isaac_v1"], metadata_path)
    frame = pd.read_csv(metadata_path, keep_default_na=False)
    unique = frame.loc[:, ["video", "action"]].drop_duplicates(ignore_index=True)
    failures: list[dict[str, str]] = []
    checked = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _audit_one, data_root, row.video, row.action
            ): (row.video, row.action)
            for row in unique.itertuples(index=False)
        }
        for future in concurrent.futures.as_completed(futures):
            video, action = futures[future]
            try:
                future.result()
                checked += 1
            except Exception as exc:
                failures.append(
                    {"video": video, "action": action, "error": str(exc)}
                )
            done = checked + len(failures)
            if done % 250 == 0 or done == len(unique):
                print(
                    f"[isaac-preflight] {done}/{len(unique)} "
                    f"failures={len(failures)}",
                    flush=True,
                )

    pair_counts = Counter(zip(frame["video"], frame["action"]))
    report = {
        "schema_version": 1,
        "metadata": metadata.to_dict(),
        "dataset_root": str(data_root),
        "rows": len(frame),
        "unique_clips": len(unique),
        "duplicate_rows": len(frame) - len(unique),
        "max_repeat_count": max(pair_counts.values()),
        "checked_unique_clips": checked,
        "video_contract": {
            "height": 480,
            "width": 832,
            "num_frames": 101,
            "fps": 20,
        },
        "action_contract": {
            "rows": 101,
            "columns": 8,
            "binary": True,
        },
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_write_json(report_path, report)
    if failures:
        raise RuntimeError(
            f"Isaac data preflight found {len(failures)} invalid clips; "
            f"see {report_path}"
        )

    smoke_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[:16].to_csv(smoke_metadata_path, index=False)
    print(f"[isaac-preflight] wrote {report_path}", flush=True)
    print(f"[isaac-preflight] wrote {smoke_metadata_path}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=ISAAC_DATA_ROOT)
    parser.add_argument("--metadata", type=Path, default=ISAAC_METADATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--smoke_metadata", type=Path, default=DEFAULT_SMOKE_METADATA
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    run_preflight(
        data_root=args.data_root.resolve(),
        metadata_path=args.metadata.resolve(),
        report_path=args.report.resolve(),
        smoke_metadata_path=args.smoke_metadata.resolve(),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
