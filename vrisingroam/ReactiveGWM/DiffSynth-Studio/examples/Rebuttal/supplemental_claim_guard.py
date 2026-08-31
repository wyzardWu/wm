#!/usr/bin/env python3
"""Reserve a lower CSV range for already-running canonical cache workers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REACTIVE_ROOT = REPO_ROOT / "examples/ReactiveGWM"
REACTIVE_SCRIPTS = REACTIVE_ROOT / "scripts"
for path in (REACTIVE_ROOT, REACTIVE_SCRIPTS, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.ReactiveGWM.scripts.precompute_cache import (  # noqa: E402
    _shard_path,
    video_cache_key,
)
from examples.Rebuttal.supplemental_vae_worker import (  # noqa: E402
    claim_path,
    try_acquire_claim,
)


def install_boundary_guards(
    *,
    metadata: str | Path,
    cache_root: str | Path,
    below_csv_index: int,
    height: int = 480,
    width: int = 608,
    num_frames: int = 101,
) -> dict[str, Any]:
    """Claim every incomplete row below the boundary for canonical workers."""

    frame = pd.read_csv(metadata, keep_default_na=False)
    if not 0 <= below_csv_index <= len(frame):
        raise ValueError(
            f"below_csv_index must be in [0, {len(frame)}], " f"got {below_csv_index}"
        )
    root = Path(cache_root)
    created = 0
    already_claimed = 0
    already_complete = 0
    for csv_index in range(below_csv_index):
        rel_video = frame.iloc[csv_index]["video"]
        video_hash = video_cache_key(
            rel_video,
            height,
            width,
            num_frames,
            16,
            16,
            False,
        )
        first_frame_hash = video_cache_key(
            rel_video,
            height,
            width,
            num_frames,
            16,
            16,
            True,
        )
        video_path = _shard_path(root, "video", video_hash)
        first_frame_path = _shard_path(root, "first_frame", first_frame_hash)
        if video_path.is_file() and first_frame_path.is_file():
            already_complete += 1
            continue
        acquired = try_acquire_claim(
            claim_path(root, video_hash),
            {
                "kind": "canonical_boundary_guard",
                "csv_index": csv_index,
                "rel_video": rel_video,
                "below_csv_index": below_csv_index,
            },
            stale_after_seconds=10**12,
        )
        if acquired:
            created += 1
        else:
            already_claimed += 1
    return {
        "metadata": str(Path(metadata).resolve()),
        "cache_root": str(root.resolve()),
        "below_csv_index": below_csv_index,
        "created": created,
        "already_claimed": already_claimed,
        "already_complete": already_complete,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--below_csv_index", required=True, type=int)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--num_frames", type=int, default=101)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = install_boundary_guards(
        metadata=args.metadata,
        cache_root=args.cache_root,
        below_csv_index=args.below_csv_index,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    print(report)


if __name__ == "__main__":
    main()
