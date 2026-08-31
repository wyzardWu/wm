"""Read-only preflight validation for rebuttal data, cache, and resources."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .checkpoint_io import sha256_file
from .prepare_metadata import MetadataValidationError, extract_strategy
from .variants import (
    FORMAL_DEFAULTS,
    FORMAL_NUM_PROCESSES,
    METADATA_MANIFEST,
    VariantSpec,
    wan_model_paths_json,
    formal_defaults_for,
)


@dataclass(frozen=True)
class MetadataAudit:
    path: str
    sha256: str
    md5: str
    rows: int
    prompt_mode: str
    first_video: str
    first_action: str
    exact_manifest_match: bool
    unique_clips: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheAudit:
    root: str
    manifest: str
    rows: int
    video_dir: str
    first_frame_dir: str
    t5_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def md5_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_metadata_hash(
    spec: VariantSpec,
    metadata_path: Path,
    manifest_path: Path,
) -> tuple[str | None, bool]:
    if not manifest_path.is_file():
        return None, False
    if spec.is_isaac:
        return None, metadata_path.resolve() == spec.metadata_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec.key in {"v1", "v2"}:
        entry = manifest["outputs"][spec.key]
    else:
        entry = manifest["sources"]["structured"]
    expected_path = Path(entry["path"]).resolve()
    exact_path = metadata_path.resolve() == expected_path
    return (entry["sha256"] if exact_path else None), exact_path


def _validate_prompt(prompt: str, mode: str, row_number: int) -> None:
    if mode == "isaac_vanilla_strategy":
        if not prompt.strip():
            raise MetadataValidationError(f"row {row_number}: blank Isaac prompt")
        return
    try:
        strategy = extract_strategy(prompt)
    except MetadataValidationError as exc:
        raise MetadataValidationError(f"row {row_number}: {exc}") from exc

    if mode == "strategy_only":
        if prompt != strategy:
            raise MetadataValidationError(
                f"row {row_number}: V2 prompt is not exactly Strategy(...)"
            )
        return

    if mode == "vanilla_strategy":
        prefix = prompt[: -len(strategy)]
        vanilla = prefix[:-1] if prefix.endswith(" ") else ""
        if (
            not vanilla
            or not prefix.endswith(" ")
            or vanilla[-1].isspace()
            or "Strategy(" in vanilla
        ):
            raise MetadataValidationError(
                f"row {row_number}: V1 must be vanilla.rstrip() + one ASCII "
                "space + Strategy(...)"
            )
        return

    if mode == "structured":
        required = (
            "NPC: Active_Behavior(",
            "Passive_Behavior(",
            "Strategy(",
        )
        if not all(marker in prompt for marker in required):
            raise MetadataValidationError(
                f"row {row_number}: V3 lacks full Active/Passive/Strategy structure"
            )
        return

    raise ValueError(f"Unknown prompt mode: {mode}")


def validate_metadata(
    spec: VariantSpec,
    metadata_path: str | Path,
    *,
    manifest_path: str | Path = METADATA_MANIFEST,
    allow_isaac_subset: bool = False,
) -> MetadataAudit:
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata is missing: {path}")

    seen: set[tuple[str, str]] = set()
    row_count = 0
    first_video = ""
    first_action = ""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"video", "action", "prompt"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise MetadataValidationError(
                f"Metadata lacks columns {sorted(missing)}: {path}"
            )
        for row_number, row in enumerate(reader, start=2):
            values = {key: value for key, value in row.items() if key in required}
            blank = [
                key for key, value in values.items() if not value or not value.strip()
            ]
            if blank:
                raise MetadataValidationError(f"row {row_number}: blank fields {blank}")
            key = (row["video"], row["action"])
            if key in seen and not spec.is_isaac:
                raise MetadataValidationError(
                    f"row {row_number}: duplicate (video, action)={key}"
                )
            seen.add(key)
            _validate_prompt(row["prompt"], spec.prompt_mode, row_number)
            if row_count == 0:
                first_video, first_action = key
            row_count += 1
    if row_count == 0:
        raise MetadataValidationError(f"Metadata has no rows: {path}")
    if spec.is_isaac and not allow_isaac_subset:
        if row_count != 5_000:
            raise MetadataValidationError(
                f"Isaac metadata must have exactly 5000 rows, got {row_count}"
            )
        if len(seen) != 4_119:
            raise MetadataValidationError(
                "Isaac metadata must have exactly 4119 unique (video, action) "
                f"clips, got {len(seen)}"
            )

    actual_sha = sha256_file(path)
    expected_sha, exact_path = _expected_metadata_hash(spec, path, Path(manifest_path))
    if spec.is_isaac and allow_isaac_subset:
        expected_sha, exact_path = None, False
    if expected_sha is not None and actual_sha != expected_sha:
        raise MetadataValidationError(
            f"Metadata SHA256 differs from generated manifest: "
            f"actual={actual_sha}, expected={expected_sha}"
        )
    return MetadataAudit(
        path=str(path.resolve()),
        sha256=actual_sha,
        md5=md5_file(path),
        rows=row_count,
        prompt_mode=spec.prompt_mode,
        first_video=first_video,
        first_action=first_action,
        exact_manifest_match=bool(exact_path and expected_sha == actual_sha),
        unique_clips=len(seen),
    )


def validate_cache(
    cache_root: str | Path,
    metadata: MetadataAudit,
    *,
    height: int,
    width: int,
    num_frames: int,
    variant: str | None = None,
) -> CacheAudit:
    root = Path(cache_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("csv_md5") != metadata.md5:
        raise ValueError(
            "Cache metadata hash mismatch: "
            f"cache={manifest.get('csv_md5')}, training={metadata.md5}"
        )
    rebuttal = manifest.get("rebuttal") or {}
    if variant is not None and rebuttal.get("variant") != variant:
        raise ValueError(
            f"Cache variant mismatch: cache={rebuttal.get('variant')}, "
            f"training={variant}"
        )
    if rebuttal.get("metadata_sha256") != metadata.sha256:
        raise ValueError(
            "Cache SHA256 metadata fingerprint mismatch: "
            f"cache={rebuttal.get('metadata_sha256')}, "
            f"training={metadata.sha256}"
        )
    if int(manifest.get("num_rows", -1)) != metadata.rows:
        raise ValueError(
            "Cache row count mismatch: "
            f"cache={manifest.get('num_rows')}, training={metadata.rows}"
        )
    config = manifest.get("config") or {}
    expected = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "height_division_factor": 16,
        "width_division_factor": 16,
        "use_csv_prompt": True,
        "prompt_column": "prompt",
    }
    mismatches = {
        key: {"cache": config.get(key), "training": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Cache configuration mismatch: {mismatches}")
    if variant == "isaac_v1":
        from .isaac_profile import ISAAC_ACTION_COLUMNS, ISAAC_ACTION_INDICES

        isaac_expected = {
            "unique_clips": 4_119,
            "action_columns": list(ISAAC_ACTION_COLUMNS),
            "action_dim": 8,
            "action_hold_window": 1,
            "action_temporal_mapping": "causal_first_4n1",
            "action_indices": list(ISAAC_ACTION_INDICES),
        }
        isaac_mismatches = {
            key: {"cache": rebuttal.get(key), "training": value}
            for key, value in isaac_expected.items()
            if rebuttal.get(key) != value
        }
        if isaac_mismatches:
            raise ValueError(f"Isaac cache contract mismatch: {isaac_mismatches}")
    failed = manifest.get("failed_rows") or []
    if failed:
        raise ValueError(f"Cache manifest records {len(failed)} failed rows")
    if len(manifest.get("rows") or []) != metadata.rows:
        raise ValueError("Cache manifest rows table is incomplete")

    directories = {
        kind: (root / kind).resolve() for kind in ("video", "first_frame", "t5")
    }
    missing_dirs = [kind for kind, path in directories.items() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Cache shard directories are missing: {missing_dirs} under {root}"
        )
    return CacheAudit(
        root=str(root.resolve()),
        manifest=str(manifest_path.resolve()),
        rows=metadata.rows,
        video_dir=str(directories["video"]),
        first_frame_dir=str(directories["first_frame"]),
        t5_dir=str(directories["t5"]),
    )


def validate_model_assets(
    wan_root: str | Path,
    tokenizer_path: str | Path,
) -> list[str]:
    root = Path(wan_root)
    tokenizer = Path(tokenizer_path)
    model_paths = json.loads(wan_model_paths_json(root))
    required = [Path(path) for path in model_paths[0]]
    required += [Path(model_paths[1]), Path(model_paths[2]), tokenizer]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required model assets are missing: " + ", ".join(missing)
        )
    return [str(path.resolve()) for path in required]


def visible_cuda_devices() -> tuple[str, ...]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def validate_formal_hyperparameters(args) -> None:
    """Fail if a formal run silently drifts from the user-approved recipe."""

    mapping = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_steps": args.max_train_steps,
        "save_steps": args.save_steps,
        "prompt_dropout_prob": args.prompt_dropout_prob,
        "action_dropout_prob": args.action_dropout_prob,
        "dataset_repeat": args.dataset_repeat,
        "dataset_num_workers": args.dataset_num_workers,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "action_hold_window": args.action_hold_window,
    }
    approved_defaults = formal_defaults_for(args.variant)
    mismatches = {
        key: {"actual": actual, "approved": approved_defaults[key]}
        for key, actual in mapping.items()
        if actual != approved_defaults[key]
    }
    if mismatches:
        raise ValueError(
            "Formal configuration differs from the approved recipe; "
            "use --smoke_test only for short validation runs. "
            f"Mismatches: {mismatches}"
        )
    approved_processes = FORMAL_NUM_PROCESSES[args.variant]
    if args.expected_num_processes != approved_processes:
        raise ValueError(
            "Formal process count differs from the approved variant recipe; "
            f"{args.variant} requires {approved_processes}, got "
            f"{args.expected_num_processes}"
        )


__all__ = [
    "CacheAudit",
    "MetadataAudit",
    "md5_file",
    "validate_cache",
    "validate_formal_hyperparameters",
    "validate_metadata",
    "validate_model_assets",
    "visible_cuda_devices",
]
