#!/usr/bin/env python3
"""Prepare a provenance-locked SF3 full300 manifest for SF2 LoRA transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import pandas as pd
from safetensors import safe_open


CATEGORIES = ("Control", "Defense", "Offense")
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"
TARGET_RE = re.compile(r"^blocks\.(\d+)\.cross_attn\.(q|k|v|o)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inspect_transfer_shapes(vanilla: Path, lora: Path) -> dict:
    with safe_open(vanilla, framework="pt", device="cpu") as handle:
        vanilla_keys = set(handle.keys())
        vanilla_shapes = {
            key: tuple(handle.get_slice(key).get_shape()) for key in vanilla_keys
        }
    with safe_open(lora, framework="pt", device="cpu") as handle:
        lora_keys = set(handle.keys())
        lora_shapes = {
            key: tuple(handle.get_slice(key).get_shape()) for key in lora_keys
        }

    if len(vanilla_keys) != 855:
        raise ValueError(
            f"SF3 Vanilla must contain 855 tensors, got {len(vanilla_keys)}"
        )
    if len(lora_keys) != 240:
        raise ValueError(f"SF2 V3 LoRA must contain 240 tensors, got {len(lora_keys)}")

    prefixes = sorted(
        key.removesuffix(LORA_A_SUFFIX)
        for key in lora_keys
        if key.endswith(LORA_A_SUFFIX)
    )
    if len(prefixes) != 120:
        raise ValueError(f"SF2 V3 LoRA must contain 120 A tensors, got {len(prefixes)}")

    ranks: set[int] = set()
    failures: list[str] = []
    targets_by_projection = {name: 0 for name in ("q", "k", "v", "o")}
    blocks: set[int] = set()
    for prefix in prefixes:
        match = TARGET_RE.fullmatch(prefix)
        if match is None:
            failures.append(f"unexpected target:{prefix}")
            continue
        block = int(match.group(1))
        projection = match.group(2)
        blocks.add(block)
        targets_by_projection[projection] += 1

        a_key = prefix + LORA_A_SUFFIX
        b_key = prefix + LORA_B_SUFFIX
        base_key = prefix + ".weight"
        if b_key not in lora_shapes:
            failures.append(f"missing B:{prefix}")
            continue
        if base_key not in vanilla_shapes:
            failures.append(f"missing SF3 base weight:{prefix}")
            continue
        a_shape = lora_shapes[a_key]
        b_shape = lora_shapes[b_key]
        base_shape = vanilla_shapes[base_key]
        if len(a_shape) != 2 or len(b_shape) != 2:
            failures.append(f"non-matrix LoRA:{prefix}:A={a_shape}:B={b_shape}")
            continue
        ranks.add(a_shape[0])
        if b_shape[1] != a_shape[0] or (b_shape[0], a_shape[1]) != base_shape:
            failures.append(
                f"shape mismatch:{prefix}:A={a_shape}:B={b_shape}:base={base_shape}"
            )
    if failures:
        raise ValueError(
            "LoRA transfer compatibility failed: " + "; ".join(failures[:8])
        )
    if blocks != set(range(30)):
        raise ValueError(f"LoRA blocks must be 0..29, got {sorted(blocks)}")
    if targets_by_projection != {"q": 30, "k": 30, "v": 30, "o": 30}:
        raise ValueError(f"LoRA projection counts mismatch: {targets_by_projection}")
    if ranks != {32}:
        raise ValueError(f"LoRA rank must be 32, got {sorted(ranks)}")

    return {
        "vanilla_tensors": len(vanilla_keys),
        "lora_tensors": len(lora_keys),
        "lora_pairs": len(prefixes),
        "lora_rank": 32,
        "blocks": 30,
        "targets_by_projection": targets_by_projection,
        "missing_targets": 0,
        "shape_mismatches": 0,
    }


def resolve_manifest(source_csv: Path, benchmark_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_csv)
    required = {
        "run_id",
        "category",
        "first_frame_png",
        "actions",
        "prompt",
        "strategy",
        "num_frames",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"full300 manifest lacks columns: {missing_columns}")
    if len(frame) != 300 or frame["run_id"].nunique() != 300:
        raise ValueError(
            f"full300 must contain 300 unique run_ids, got rows={len(frame)} "
            f"unique={frame['run_id'].nunique()}"
        )
    counts = frame["category"].value_counts().to_dict()
    expected_counts = {category: 100 for category in CATEGORIES}
    if counts != expected_counts:
        raise ValueError(
            f"category counts mismatch: got={counts}, expected={expected_counts}"
        )
    if set(frame["num_frames"].astype(int)) != {101}:
        raise ValueError(
            f"num_frames must be 101: {sorted(frame['num_frames'].unique())}"
        )

    resolved = frame.copy()
    first_frames = [
        (benchmark_root / "data" / "dim_b" / "first_frames" / f"{run_id}.png").resolve()
        for run_id in resolved["run_id"].astype(str)
    ]
    missing_first_frames = [str(path) for path in first_frames if not path.is_file()]
    missing_actions = [
        str(path)
        for path in map(Path, resolved["actions"].astype(str))
        if not path.is_file()
    ]
    if missing_first_frames or missing_actions:
        raise FileNotFoundError(
            "resolved input files are missing: "
            f"first_frames={missing_first_frames[:4]}, actions={missing_actions[:4]}"
        )
    resolved["first_frame_png"] = [str(path) for path in first_frames]
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--lora-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lora-scale", type=float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_csv = args.source_csv.resolve()
    benchmark_root = args.benchmark_root.resolve()
    vanilla = args.vanilla.resolve()
    lora = args.lora.resolve()
    lora_manifest_path = args.lora_manifest.resolve()
    run_root = args.run_root.resolve()
    for path in (source_csv, vanilla, lora, lora_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.lora_scale <= 0:
        raise ValueError("lora-scale must be positive")

    compatibility = inspect_transfer_shapes(vanilla, lora)
    resolved = resolve_manifest(source_csv, benchmark_root)
    smoke = pd.concat(
        [
            resolved.loc[resolved["category"] == category].head(1)
            for category in CATEGORIES
        ],
        ignore_index=True,
    )

    manifest_dir = run_root / "manifests"
    full_path = manifest_dir / "full300.resolved.csv"
    smoke_path = manifest_dir / "smoke3.resolved.csv"
    atomic_write_csv(resolved, full_path)
    atomic_write_csv(smoke, smoke_path)

    lora_digest = sha256_file(lora)
    lora_manifest = json.loads(lora_manifest_path.read_text(encoding="utf-8"))
    recorded_lora = (lora_manifest.get("files") or {}).get("lora") or {}
    if recorded_lora.get("sha256") != lora_digest:
        raise ValueError(
            f"LoRA SHA256 mismatch: got={lora_digest}, "
            f"manifest={recorded_lora.get('sha256')}"
        )
    if (lora_manifest.get("lora") or {}).get("rank") != 32:
        raise ValueError("source V3 manifest does not declare rank 32")

    provenance = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": "sf2_v3_cross_lora_transfer_to_sf3_vanilla",
        "composition": {
            "base": "SF3 Vanilla",
            "overlay": "SF2 V3 step-30000 cross-attention LoRA only",
            "uses_v3_full_delta": False,
            "inference_lora_scale": args.lora_scale,
        },
        "weights": {
            "vanilla": {
                "path": str(vanilla),
                "bytes": vanilla.stat().st_size,
                "sha256": sha256_file(vanilla),
            },
            "lora": {
                "path": str(lora),
                "bytes": lora.stat().st_size,
                "sha256": lora_digest,
            },
            "lora_manifest": {
                "path": str(lora_manifest_path),
                "sha256": sha256_file(lora_manifest_path),
            },
        },
        "compatibility": compatibility,
        "data": {
            "source_csv": str(source_csv),
            "source_csv_sha256": sha256_file(source_csv),
            "resolved_csv": str(full_path),
            "resolved_csv_sha256": sha256_file(full_path),
            "smoke_csv": str(smoke_path),
            "smoke_csv_sha256": sha256_file(smoke_path),
            "rows": len(resolved),
            "category_counts": {category: 100 for category in CATEGORIES},
        },
        "inference": {
            "height": 480,
            "width": 832,
            "num_frames": 101,
            "fps": 20,
            "num_inference_steps": 30,
            "cfg_scale": 5.0,
            "action_cfg_scale": 1.0,
            "action_hold_window": 10,
            "seed": 0,
        },
    }
    provenance_path = manifest_dir / "provenance.json"
    atomic_write_json(provenance, provenance_path)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
