#!/usr/bin/env python3
"""One sharded GPU worker for rebuttal VAE/T5 cache precomputation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REACTIVE_ROOT = REPO_ROOT / "examples/ReactiveGWM"
REACTIVE_SCRIPTS = REACTIVE_ROOT / "scripts"
for path in (REACTIVE_ROOT, REACTIVE_SCRIPTS, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.profiles import get_profile  # noqa: E402
from data.prompt_utils import resolve_prompt  # noqa: E402

from examples.ReactiveGWM.scripts.precompute_cache import (  # noqa: E402
    _shard_path,
    atomic_save,
    encode_prompt_bitexact,
    load_pipeline,
    phase_vae,
    t5_cache_key,
)
from examples.Rebuttal.atomic_runtime import atomic_runtime_json  # noqa: E402
from examples.Rebuttal.isaac_profile import ISAAC_PROFILE  # noqa: E402


def phase_t5_sharded(
    pipe,
    frame,
    profile,
    cache_root: Path,
    *,
    rank: int,
    world_size: int,
    skip_existing: bool,
) -> int:
    resolved = {
        resolve_prompt(frame.iloc[index].to_dict(), profile, True, "prompt")
        for index in range(len(frame))
    }
    unique = sorted(resolved | {""})
    selected = [
        prompt for index, prompt in enumerate(unique) if index % world_size == rank
    ]
    print(
        f"[T5] rank={rank}/{world_size}, unique={len(unique)}, "
        f"assigned={len(selected)}",
        flush=True,
    )
    written = 0
    started = time.time()
    for index, prompt in enumerate(selected, start=1):
        digest = t5_cache_key(prompt)
        destination = _shard_path(cache_root, "t5", digest)
        if skip_existing and destination.is_file():
            continue
        with torch.no_grad():
            embedding = encode_prompt_bitexact(pipe, prompt)
        atomic_save(destination, embedding.to(dtype=pipe.torch_dtype).cpu())
        written += 1
        if index % 200 == 0 or index == len(selected):
            print(
                f"[T5] rank={rank}: {index}/{len(selected)} "
                f"[{time.time() - started:.1f}s]",
                flush=True,
            )
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--dataset_base", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--model_paths", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world_size", required=True, type=int)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--num_frames", type=int, default=101)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--t5_only",
        action="store_true",
        help=(
            "Encode only this rank's T5 prompt shard, then exit without "
            "constructing video operators or touching the shared VAE cache."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise SystemExit(
            f"Invalid shard rank/world_size: {args.rank}/{args.world_size}"
        )
    profile = ISAAC_PROFILE if args.variant == "isaac_v1" else get_profile("sf2")
    cache_root = Path(args.cache_root)
    frame = pd.read_csv(args.metadata, keep_default_na=False)
    frame = frame.assign(_csv_index=frame.index.values)
    shard = frame[frame["_csv_index"] % args.world_size == args.rank].reset_index(
        drop=True
    )

    pipe = load_pipeline(
        args.model_paths,
        args.tokenizer_path,
        "cuda",
        torch.bfloat16,
    )
    written_t5 = phase_t5_sharded(
        pipe,
        frame,
        profile,
        cache_root,
        rank=args.rank,
        world_size=args.world_size,
        skip_existing=args.skip_existing,
    )
    pipe.text_encoder = None
    torch.cuda.empty_cache()

    if args.t5_only:
        report = {
            "rank": args.rank,
            "world_size": args.world_size,
            "metadata": str(Path(args.metadata).resolve()),
            "mode": "t5_only",
            "rows_assigned": 0,
            "t5_written": written_t5,
            "failed_rows": [],
        }
        report_path = cache_root / "_workers" / f"t5-rank-{args.rank}.json"
        atomic_runtime_json(report_path, report)
        print(json.dumps(report, indent=2), flush=True)
        return

    video_op = __import__(
        "diffsynth.core", fromlist=["UnifiedDataset"]
    ).UnifiedDataset.default_video_operator(
        base_path=args.dataset_base,
        max_pixels=1920 * 1080,
        height=args.height,
        width=args.width,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=args.num_frames,
        time_division_factor=4,
        time_division_remainder=1,
    )
    _, failed = phase_vae(
        pipe,
        shard,
        profile,
        video_op,
        cache_root,
        args.height,
        args.width,
        args.num_frames,
        16,
        16,
        True,
        "prompt",
        args.skip_existing,
    )
    report = {
        "rank": args.rank,
        "world_size": args.world_size,
        "metadata": str(Path(args.metadata).resolve()),
        "rows_assigned": len(shard),
        "t5_written": written_t5,
        "failed_rows": failed,
    }
    report_path = cache_root / "_workers" / f"rank-{args.rank}.json"
    atomic_runtime_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    if failed:
        raise SystemExit(f"Cache worker rank {args.rank} had {len(failed)} failures")


if __name__ == "__main__":
    main()
