"""Shared-VAE / per-variant-T5 cache layout and manifest finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REACTIVE_ROOT = REPO_ROOT / "examples/ReactiveGWM"
if str(REACTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(REACTIVE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.profiles import get_profile  # noqa: E402
from data.prompt_utils import resolve_prompt  # noqa: E402

from examples.ReactiveGWM.model_training.cached_dataset import (  # noqa: E402
    _shard_path,
    t5_cache_key,
    video_cache_key,
)
from examples.Rebuttal.checkpoint_io import atomic_write_json, sha256_file  # noqa: E402
from examples.Rebuttal.preflight import md5_file  # noqa: E402
from examples.Rebuttal.variants import resolve_variant  # noqa: E402
from examples.Rebuttal.isaac_profile import (  # noqa: E402
    ISAAC_ACTION_COLUMNS,
    ISAAC_ACTION_INDICES,
    ISAAC_PROFILE,
)


CACHE_MANIFEST_VERSION = 1
VARIANT_KEYS = ("v1", "v2", "v3")
FINALIZE_VARIANT_KEYS = (*VARIANT_KEYS, "isaac_v1")


def variant_cache_root(cache_base: str | Path, variant: str) -> Path:
    spec = resolve_variant(variant)
    if spec.is_isaac:
        return Path(cache_base)
    return Path(cache_base) / spec.key


def _ensure_symlink(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(
                f"Existing symlink points elsewhere: {link} -> {link.resolve()}, "
                f"expected {target.resolve()}"
            )
        return
    if link.exists():
        raise RuntimeError(
            f"Refusing to replace existing non-symlink cache path: {link}"
        )
    link.symlink_to(target.resolve(), target_is_directory=True)


def initialize_layout(cache_base: str | Path) -> dict[str, Any]:
    base = Path(cache_base).resolve()
    shared = base / "shared_vae"
    video = shared / "video"
    first_frame = shared / "first_frame"
    video.mkdir(parents=True, exist_ok=True)
    first_frame.mkdir(parents=True, exist_ok=True)

    variants = {}
    for key in VARIANT_KEYS:
        root = base / key
        root.mkdir(parents=True, exist_ok=True)
        (root / "t5").mkdir(exist_ok=True)
        _ensure_symlink(root / "video", video)
        _ensure_symlink(root / "first_frame", first_frame)
        variants[key] = {
            "root": str(root),
            "video": str((root / "video").resolve()),
            "first_frame": str((root / "first_frame").resolve()),
            "t5": str((root / "t5").resolve()),
        }
    payload = {
        "schema_version": 1,
        "shared_vae": {
            "video": str(video.resolve()),
            "first_frame": str(first_frame.resolve()),
        },
        "variants": variants,
    }
    atomic_write_json(base / "layout.json", payload)
    return payload


def _file_prefix_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1 << 20))
    return digest.hexdigest()[:16]


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def finalize_manifest(
    *,
    cache_base: str | Path,
    variant: str,
    metadata_path: str | Path,
    dataset_base: str | Path,
    vae_path: str | Path,
    t5_path: str | Path,
    height: int = 480,
    width: int = 608,
    num_frames: int = 101,
    height_division_factor: int = 16,
    width_division_factor: int = 16,
    max_pixels: int = 1920 * 1080,
) -> dict[str, Any]:
    spec = resolve_variant(variant)
    root = variant_cache_root(cache_base, spec.key)
    profile = ISAAC_PROFILE if spec.is_isaac else get_profile("sf2")
    metadata = Path(metadata_path).resolve()
    frame = pd.read_csv(metadata, keep_default_na=False)
    if len(frame) == 0:
        raise ValueError(f"Cannot finalize an empty cache: {metadata}")

    rows = []
    missing = []
    for csv_index, row in frame.iterrows():
        record = row.to_dict()
        rel_video = record["video"]
        video_hash = video_cache_key(
            rel_video,
            height,
            width,
            num_frames,
            height_division_factor,
            width_division_factor,
            False,
        )
        first_frame_hash = video_cache_key(
            rel_video,
            height,
            width,
            num_frames,
            height_division_factor,
            width_division_factor,
            True,
        )
        prompt_hash = t5_cache_key(resolve_prompt(record, profile, True, "prompt"))
        row_meta = {
            "csv_index": int(csv_index),
            "rel_video": rel_video,
            "video_hash": video_hash,
            "first_frame_hash": first_frame_hash,
            "prompt_hash": prompt_hash,
        }
        rows.append(row_meta)
        for kind, digest in (
            ("video", video_hash),
            ("first_frame", first_frame_hash),
            ("t5", prompt_hash),
        ):
            path = _shard_path(root, kind, digest)
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break

    empty_prompt_hash = t5_cache_key("")
    empty_path = _shard_path(root, "t5", empty_prompt_hash)
    if not empty_path.is_file():
        missing.append(str(empty_path))
    if missing:
        raise FileNotFoundError(
            "Cache is incomplete; first missing shards:\n  " + "\n  ".join(missing[:20])
        )

    example = rows[0]
    video_tensor = _load_tensor(_shard_path(root, "video", example["video_hash"]))
    first_frame_tensor = _load_tensor(
        _shard_path(root, "first_frame", example["first_frame_hash"])
    )
    if video_tensor.ndim != 5 or first_frame_tensor.ndim != 5:
        raise ValueError(
            f"Unexpected cache tensor ranks: video={video_tensor.shape}, "
            f"first_frame={first_frame_tensor.shape}"
        )
    if height % video_tensor.shape[-2] or width % video_tensor.shape[-1]:
        raise ValueError(
            f"Cannot derive VAE scale from cached shape {video_tensor.shape}"
        )
    up_h = height // video_tensor.shape[-2]
    up_w = width // video_tensor.shape[-1]
    if up_h != up_w:
        raise ValueError(f"VAE spatial scales differ: H={up_h}, W={up_w}")

    vae = Path(vae_path).resolve()
    t5 = Path(t5_path).resolve()
    for path in (vae, t5):
        if not path.is_file():
            raise FileNotFoundError(f"Model fingerprint source missing: {path}")
    manifest = {
        "version": CACHE_MANIFEST_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv_path": str(metadata),
        "csv_md5": md5_file(metadata),
        "num_rows": int(len(frame)),
        "dataset_base": str(Path(dataset_base).resolve()),
        "game": profile.name,
        "config": {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "height_division_factor": height_division_factor,
            "width_division_factor": width_division_factor,
            "max_pixels": max_pixels,
            "vae_upsampling_factor": up_h,
            "vae_z_dim": int(video_tensor.shape[1]),
            "latent_shape": list(video_tensor.shape),
            "first_frame_latent_shape": list(first_frame_tensor.shape),
            "cache_dtype": str(video_tensor.dtype).removeprefix("torch."),
            "use_csv_prompt": True,
            "prompt_column": "prompt",
            "fixed_prompt": profile.fixed_prompt,
        },
        "model_fingerprints": {
            "vae_path": str(vae),
            "vae_sha256_prefix": _file_prefix_fingerprint(vae),
            "t5_path": str(t5),
            "t5_sha256_prefix": _file_prefix_fingerprint(t5),
        },
        "rows": rows,
        "prompt_table": {},
        "empty_prompt_hash": empty_prompt_hash,
        "failed_rows": [],
        "rebuttal": {
            "variant": spec.key,
            "prompt_mode": spec.prompt_mode,
            "metadata_sha256": sha256_file(metadata),
            "shared_video_dir": str((root / "video").resolve()),
            "shared_first_frame_dir": str((root / "first_frame").resolve()),
            "variant_t5_dir": str((root / "t5").resolve()),
            **(
                {
                    "unique_clips": int(
                        frame.loc[:, ["video", "action"]].drop_duplicates().shape[0]
                    ),
                    "action_columns": list(ISAAC_ACTION_COLUMNS),
                    "action_dim": len(ISAAC_ACTION_COLUMNS),
                    "action_hold_window": 1,
                    "action_temporal_mapping": "causal_first_4n1",
                    "action_indices": list(ISAAC_ACTION_INDICES),
                }
                if spec.is_isaac
                else {}
            ),
        },
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--cache_base", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--cache_base", required=True)
    finalize.add_argument("--variant", required=True, choices=FINALIZE_VARIANT_KEYS)
    finalize.add_argument("--metadata", required=True)
    finalize.add_argument("--dataset_base", required=True)
    finalize.add_argument("--vae_path", required=True)
    finalize.add_argument("--t5_path", required=True)
    finalize.add_argument("--height", type=int, default=480)
    finalize.add_argument("--width", type=int, default=608)
    finalize.add_argument("--num_frames", type=int, default=101)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init":
        result = initialize_layout(args.cache_base)
    else:
        result = finalize_manifest(
            cache_base=args.cache_base,
            variant=args.variant,
            metadata_path=args.metadata,
            dataset_base=args.dataset_base,
            vae_path=args.vae_path,
            t5_path=args.t5_path,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
