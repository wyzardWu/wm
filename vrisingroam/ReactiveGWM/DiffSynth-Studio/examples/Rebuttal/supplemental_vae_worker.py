#!/usr/bin/env python3
"""Best-effort VAE-only cache worker that can join an active cache run."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REACTIVE_ROOT = REPO_ROOT / "examples/ReactiveGWM"
REACTIVE_SCRIPTS = REACTIVE_ROOT / "scripts"
for path in (REACTIVE_ROOT, REACTIVE_SCRIPTS, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.ReactiveGWM.scripts.precompute_cache import (  # noqa: E402
    _shard_path,
    atomic_save,
    encode_first_frame_bitexact,
    encode_video_bitexact,
    load_pipeline,
    video_cache_key,
)
from examples.Rebuttal.atomic_runtime import atomic_runtime_json  # noqa: E402


STOP_REQUESTED = False


def request_stop(_signum: int, _frame: Any) -> None:
    """Ask the worker to exit after its current cache item is atomic."""

    global STOP_REQUESTED
    STOP_REQUESTED = True


def claim_path(cache_root: str | Path, digest: str) -> Path:
    """Return the supplemental-only claim path for one video pair."""

    return Path(cache_root) / ".supplemental_claims" / digest[:2] / f"{digest}.claim"


def try_acquire_claim(
    path: str | Path,
    payload: dict[str, Any],
    *,
    stale_after_seconds: float,
) -> bool:
    """Atomically acquire a claim, recovering only sufficiently stale claims."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        age = time.time() - destination.stat().st_mtime
    except FileNotFoundError:
        age = 0.0
    if age > stale_after_seconds:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(destination, flags, 0o664)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    return True


def release_claim(path: str | Path) -> None:
    """Release a claim owned by this worker."""

    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dataset_base", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--vae_model_paths", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--num_frames", type=int, default=101)
    parser.add_argument("--cpu_threads", type=int, default=6)
    parser.add_argument("--min_csv_index", type=int, default=0)
    parser.add_argument("--claim_ttl_seconds", type=float, default=3600.0)
    parser.add_argument("--progress_every", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu_threads must be positive")
    if args.claim_ttl_seconds <= 0:
        raise ValueError("--claim_ttl_seconds must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress_every must be positive")

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cache_root = Path(args.cache_root)
    manifest_path = cache_root / "manifest.json"
    frame = pd.read_csv(args.metadata, keep_default_na=False)
    frame = frame.assign(_csv_index=frame.index.values)
    if not 0 <= args.min_csv_index < len(frame):
        raise ValueError(
            f"--min_csv_index must be in [0, {len(frame) - 1}], "
            f"got {args.min_csv_index}"
        )

    print(
        f"[supplemental] worker={args.worker_id}, rows={len(frame)}, "
        f"direction=reverse, min_csv_index={args.min_csv_index}, "
        f"cpu_threads={args.cpu_threads}",
        flush=True,
    )
    pipe = load_pipeline(
        args.vae_model_paths,
        args.tokenizer_path,
        "cuda",
        torch.bfloat16,
    )
    if pipe.vae is None:
        raise RuntimeError("VAE-only pipeline did not load a VAE")
    if getattr(pipe, "text_encoder", None) is not None:
        raise RuntimeError("Supplemental VAE worker unexpectedly loaded T5")

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

    started = time.time()
    visited = 0
    claimed_elsewhere = 0
    rows_written = 0
    video_written = 0
    first_frame_written = 0
    failures: list[dict[str, Any]] = []
    stop_reason = "scan_complete"

    for index in range(len(frame) - 1, args.min_csv_index - 1, -1):
        if STOP_REQUESTED:
            stop_reason = "signal"
            break
        if manifest_path.is_file():
            stop_reason = "manifest_exists"
            break

        visited += 1
        row = frame.iloc[index].to_dict()
        rel_video = row["video"]
        video_hash = video_cache_key(
            rel_video,
            args.height,
            args.width,
            args.num_frames,
            16,
            16,
            False,
        )
        first_frame_hash = video_cache_key(
            rel_video,
            args.height,
            args.width,
            args.num_frames,
            16,
            16,
            True,
        )
        video_path = _shard_path(cache_root, "video", video_hash)
        first_frame_path = _shard_path(
            cache_root,
            "first_frame",
            first_frame_hash,
        )
        if video_path.is_file() and first_frame_path.is_file():
            continue

        lock_path = claim_path(cache_root, video_hash)
        acquired = try_acquire_claim(
            lock_path,
            {
                "worker_id": args.worker_id,
                "pid": os.getpid(),
                "csv_index": int(row["_csv_index"]),
                "rel_video": rel_video,
                "created_unix": time.time(),
            },
            stale_after_seconds=args.claim_ttl_seconds,
        )
        if not acquired:
            claimed_elsewhere += 1
            continue

        wrote_row = False
        try:
            if video_path.is_file() and first_frame_path.is_file():
                continue
            pil_frames = video_op(rel_video)
            if not video_path.is_file():
                with torch.no_grad():
                    latent = encode_video_bitexact(pipe, pil_frames)
                atomic_save(
                    video_path,
                    latent.to(dtype=pipe.torch_dtype).cpu(),
                )
                video_written += 1
                wrote_row = True
            if not first_frame_path.is_file():
                with torch.no_grad():
                    first_frame_latent = encode_first_frame_bitexact(
                        pipe,
                        pil_frames[0],
                        args.height,
                        args.width,
                    )
                atomic_save(
                    first_frame_path,
                    first_frame_latent.to(dtype=pipe.torch_dtype).cpu(),
                )
                first_frame_written += 1
                wrote_row = True
            if wrote_row:
                rows_written += 1
        except Exception as exc:  # noqa: BLE001
            failure = {
                "csv_index": int(row["_csv_index"]),
                "rel_video": rel_video,
                "error": repr(exc),
            }
            failures.append(failure)
            print(f"[supplemental][fail] {failure}", flush=True)
        finally:
            release_claim(lock_path)

        if rows_written and rows_written % args.progress_every == 0:
            elapsed = time.time() - started
            print(
                f"[supplemental] worker={args.worker_id}, "
                f"written={rows_written}, visited={visited}, "
                f"rate={rows_written / max(elapsed, 1e-3):.3f} r/s, "
                f"failures={len(failures)}",
                flush=True,
            )

    report = {
        "worker_id": args.worker_id,
        "pid": os.getpid(),
        "metadata": str(Path(args.metadata).resolve()),
        "cache_root": str(cache_root.resolve()),
        "direction": "reverse",
        "min_csv_index": args.min_csv_index,
        "visited": visited,
        "claimed_elsewhere": claimed_elsewhere,
        "rows_written": rows_written,
        "video_written": video_written,
        "first_frame_written": first_frame_written,
        "failures": failures,
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = cache_root / "_supplemental" / f"{args.worker_id}.json"
    atomic_runtime_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
