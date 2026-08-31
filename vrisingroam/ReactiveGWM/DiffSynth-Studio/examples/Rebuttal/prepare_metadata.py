"""Build deterministic SF2 metadata for the rebuttal prompt ablations.

The two source CSV files are treated as immutable.  Rows are joined by the
``(video, action)`` key rather than by row position, even though the currently
released files have identical ordering.

Outputs:

* V1: ``vanilla_prompt.rstrip() + " " + Strategy(...)``
* V2: ``Strategy(...)``

The script writes outputs atomically and records source/output hashes plus
validation statistics in a JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd


SCRIPT_VERSION = 1
REQUIRED_COLUMNS = ("video", "action", "prompt")
KEY_COLUMNS = ("video", "action")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(
    "/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF2"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_TOKENIZER = Path(
    "/nfs/zeqingwang/models/base_model/" "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
)


class MetadataValidationError(ValueError):
    """Raised when source metadata violates the rebuttal data contract."""


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_strategy(prompt: object) -> str:
    """Extract one balanced, terminal ``Strategy(...)`` expression.

    A small balanced-parenthesis parser is used instead of a regex so future
    descriptions may contain parenthesized text without silently truncating the
    strategy.  Extra non-whitespace text after the closing parenthesis is
    rejected.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise MetadataValidationError("prompt is missing or blank")

    marker = "Strategy("
    if prompt.count(marker) != 1:
        raise MetadataValidationError(
            f"expected exactly one {marker!r}, found {prompt.count(marker)}"
        )
    start = prompt.index(marker)
    depth = 0
    end = None
    for index in range(start + len("Strategy"), len(prompt)):
        char = prompt[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise MetadataValidationError("unbalanced Strategy parentheses")
            if depth == 0:
                end = index + 1
                break
    if end is None or depth != 0:
        raise MetadataValidationError("unterminated Strategy expression")
    if prompt[end:].strip():
        raise MetadataValidationError(
            f"unexpected trailing text after Strategy: {prompt[end:]!r}"
        )
    strategy = prompt[start:end]
    if strategy == "Strategy()":
        raise MetadataValidationError("empty Strategy expression")
    return strategy


def _read_source(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} metadata not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MetadataValidationError(
            f"{label} metadata is missing columns {missing}; "
            f"found={list(frame.columns)}"
        )
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        blank = frame[column].map(
            lambda value: not isinstance(value, str) or not value.strip()
        )
        if blank.any():
            rows = frame.index[blank].tolist()[:10]
            raise MetadataValidationError(
                f"{label} metadata has blank {column!r} at rows {rows}"
            )
    duplicated = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicated.any():
        preview = frame.loc[duplicated, list(KEY_COLUMNS)].head(10).to_dict("records")
        raise MetadataValidationError(
            f"{label} metadata has duplicate (video, action) keys: {preview}"
        )
    return frame


def build_variant_frames(
    vanilla: pd.DataFrame,
    structured: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return V1, V2, and the aligned list of extracted strategies."""

    vanilla = vanilla.copy()
    structured = structured.copy()
    vanilla["_source_order"] = range(len(vanilla))

    merged = vanilla.merge(
        structured,
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("_vanilla", "_structured"),
        indicator=True,
        validate="one_to_one",
        sort=False,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        preview = (
            merged.loc[unmatched, [*KEY_COLUMNS, "_merge"]].head(10).to_dict("records")
        )
        counts = merged.loc[unmatched, "_merge"].value_counts().to_dict()
        raise MetadataValidationError(
            f"source key sets do not match: counts={counts}, preview={preview}"
        )

    merged = merged.sort_values("_source_order", kind="stable").reset_index(drop=True)
    strategies: list[str] = []
    errors: list[str] = []
    for row_index, prompt in enumerate(merged["prompt_structured"]):
        try:
            strategies.append(extract_strategy(prompt))
        except MetadataValidationError as exc:
            errors.append(f"row={row_index}: {exc}")
            if len(errors) >= 10:
                break
    if errors:
        raise MetadataValidationError(
            "failed to extract Strategy expressions:\n  " + "\n  ".join(errors)
        )

    vanilla_prompts = merged["prompt_vanilla"].map(str)
    v1 = pd.DataFrame(
        {
            "video": merged["video"],
            "action": merged["action"],
            "prompt": [
                vanilla_prompt.rstrip() + " " + strategy
                for vanilla_prompt, strategy in zip(vanilla_prompts, strategies)
            ],
        }
    )
    v2 = pd.DataFrame(
        {
            "video": merged["video"],
            "action": merged["action"],
            "prompt": strategies,
        }
    )
    return v1, v2, strategies


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False, lineterminator="\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _token_length_summary(
    name: str,
    prompts: Sequence[str],
    tokenizer,
) -> dict:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    lengths = sorted(len(ids) for ids in encoded["input_ids"])
    if not lengths:
        raise MetadataValidationError(f"{name} contains no prompts")

    def percentile(fraction: float) -> int:
        index = min(int(len(lengths) * fraction), len(lengths) - 1)
        return lengths[index]

    summary = {
        "min": lengths[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": lengths[-1],
        "gt_512": sum(length > 512 for length in lengths),
    }
    if summary["gt_512"]:
        raise MetadataValidationError(
            f"{name} has {summary['gt_512']} prompts exceeding 512 tokens"
        )
    return summary


def _strategy_category(strategy: str) -> str:
    body = strategy[len("Strategy(") :]
    return body.split(":", 1)[0].strip()


def _preview(frame: pd.DataFrame, count: int = 2) -> list[dict]:
    if len(frame) <= count * 2:
        selected = frame
    else:
        selected = pd.concat([frame.head(count), frame.tail(count)])
    return selected.to_dict("records")


def generate_metadata(
    vanilla_path: Path,
    structured_path: Path,
    output_dir: Path,
    tokenizer_path: Path | None = DEFAULT_TOKENIZER,
    smoke_rows: int = 16,
) -> dict:
    """Generate real V1/V2 metadata and return the written manifest."""

    vanilla_path = vanilla_path.resolve()
    structured_path = structured_path.resolve()
    output_dir = output_dir.resolve()

    source_hashes_before = {
        "vanilla": sha256_file(vanilla_path),
        "structured": sha256_file(structured_path),
    }
    vanilla = _read_source(vanilla_path, "vanilla")
    structured = _read_source(structured_path, "structured")
    v1, v2, strategies = build_variant_frames(vanilla, structured)

    v1_path = output_dir / "metadata_v1_vanilla_strategy.csv"
    v2_path = output_dir / "metadata_v2_strategy_only.csv"
    _atomic_write_csv(v1, v1_path)
    _atomic_write_csv(v2, v2_path)

    smoke_outputs: dict[str, str] = {}
    if smoke_rows > 0:
        count = min(smoke_rows, len(v1))
        smoke_v1_path = output_dir / f"metadata_v1_smoke{count}.csv"
        smoke_v2_path = output_dir / f"metadata_v2_smoke{count}.csv"
        smoke_v3_path = output_dir / f"metadata_v3_smoke{count}.csv"
        _atomic_write_csv(v1.head(count), smoke_v1_path)
        _atomic_write_csv(v2.head(count), smoke_v2_path)
        _atomic_write_csv(structured.head(count), smoke_v3_path)
        smoke_outputs = {
            "v1": str(smoke_v1_path),
            "v2": str(smoke_v2_path),
            "v3": str(smoke_v3_path),
        }

    source_hashes_after = {
        "vanilla": sha256_file(vanilla_path),
        "structured": sha256_file(structured_path),
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError(
            "source metadata changed while generating outputs; refusing manifest"
        )

    token_lengths = None
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True
        )
        token_lengths = {
            "v1_vanilla_strategy": _token_length_summary(
                "v1_vanilla_strategy", v1["prompt"].tolist(), tokenizer
            ),
            "v2_strategy_only": _token_length_summary(
                "v2_strategy_only", v2["prompt"].tolist(), tokenizer
            ),
            "v3_structured": _token_length_summary(
                "v3_structured", structured["prompt"].tolist(), tokenizer
            ),
        }

    strategy_counts = Counter(strategies)
    category_counts = Counter(_strategy_category(strategy) for strategy in strategies)
    manifest = {
        "schema_version": 1,
        "generator": {
            "script": str(Path(__file__).resolve()),
            "version": SCRIPT_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "sources": {
            "vanilla": {
                "path": str(vanilla_path),
                "sha256": source_hashes_before["vanilla"],
                "rows": len(vanilla),
            },
            "structured": {
                "path": str(structured_path),
                "sha256": source_hashes_before["structured"],
                "rows": len(structured),
            },
        },
        "contract": {
            "join_keys": list(KEY_COLUMNS),
            "v1_prompt": 'vanilla_prompt.rstrip() + " " + Strategy(...)',
            "v2_prompt": "Strategy(...)",
            "action_input_preserved": True,
        },
        "outputs": {
            "v1": {
                "path": str(v1_path),
                "sha256": sha256_file(v1_path),
                "rows": len(v1),
                "preview": _preview(v1),
            },
            "v2": {
                "path": str(v2_path),
                "sha256": sha256_file(v2_path),
                "rows": len(v2),
                "preview": _preview(v2),
            },
            "smoke": smoke_outputs,
        },
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "token_lengths": token_lengths,
    }
    manifest_path = output_dir / "metadata_manifest.json"
    _atomic_write_json(manifest, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic SF2 rebuttal metadata variants."
    )
    parser.add_argument(
        "--vanilla-metadata",
        type=Path,
        default=DEFAULT_DATA_ROOT / "metadata_vanilla.csv",
    )
    parser.add_argument(
        "--structured-metadata",
        type=Path,
        default=DEFAULT_DATA_ROOT / "metadata.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Local UMT5 tokenizer. Pass --skip-tokenizer-check to avoid loading it.",
    )
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=16,
        help="Also write deterministic prefix CSVs for smoke training; 0 disables.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_rows < 0:
        raise SystemExit("--smoke-rows must be >= 0")
    manifest = generate_metadata(
        vanilla_path=args.vanilla_metadata,
        structured_path=args.structured_metadata,
        output_dir=args.output_dir,
        tokenizer_path=None if args.skip_tokenizer_check else args.tokenizer_path,
        smoke_rows=args.smoke_rows,
    )
    print(
        json.dumps(
            {
                "v1": manifest["outputs"]["v1"],
                "v2": manifest["outputs"]["v2"],
                "category_counts": manifest["category_counts"],
                "token_lengths": manifest["token_lengths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
