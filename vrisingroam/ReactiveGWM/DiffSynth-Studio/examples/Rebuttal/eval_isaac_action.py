#!/usr/bin/env python3
"""Fixed-condition single-action evaluation for the Isaac V1 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REACTIVE_ROOT = REPO_ROOT / "examples/ReactiveGWM"
for path in (REPO_ROOT, REACTIVE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffsynth.utils.data import save_video  # noqa: E402
from examples.ReactiveGWM.inference import infer_core  # noqa: E402
from examples.ReactiveGWM.inference.infer_core import CkptSpec  # noqa: E402
from examples.Rebuttal.atomic_runtime import atomic_runtime_json  # noqa: E402
from examples.Rebuttal.isaac_model import IsaacReactiveGWMModel  # noqa: E402
from examples.Rebuttal.isaac_profile import (  # noqa: E402
    ISAAC_ACTION_COLUMNS,
    ISAAC_PROFILE,
    ISAAC_RAW_FRAMES,
)
from examples.Rebuttal.variants import ISAAC_DATA_ROOT, WAN_ROOT  # noqa: E402


DEFAULT_CHECKPOINT = Path(
    "/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/"
    "Isaac_v1_vanilla_strategy/step-30000.safetensors"
)
DEFAULT_OUTPUT = Path(
    "/nfs/zeqingwang/models/train/ReactiveGWM/Rebuttal/" "Isaac_v1_action_eval"
)
DEFAULT_PROMPT = "The Binding of Isaac, in a boss room, Isaac fights Gurdy Jr."
NO_ACTION = "NO_ACTION"


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def action_names() -> tuple[str, ...]:
    return (NO_ACTION, *ISAAC_ACTION_COLUMNS)


def make_eval_action(name: str, device: str = "cuda") -> torch.Tensor:
    """Return [1, 101, 8]; frame zero is always neutral."""

    if name not in action_names():
        raise ValueError(f"Unknown Isaac action {name!r}")
    action = torch.zeros(
        (1, ISAAC_RAW_FRAMES, len(ISAAC_ACTION_COLUMNS)),
        dtype=torch.bfloat16,
        device=device,
    )
    if name != NO_ACTION:
        action[:, 1:, ISAAC_ACTION_COLUMNS.index(name)] = 1
    return action


def build_isaac_pipe(base_model_root: str, checkpoint: Path, device: str):
    """Use the Isaac DiT subclass without changing shared inference code."""

    original = infer_core.ReactiveGWMModel
    infer_core.ReactiveGWMModel = IsaacReactiveGWMModel
    try:
        return infer_core.build_pipe(
            ISAAC_PROFILE,
            base_model_root,
            CkptSpec(full_ckpt=str(checkpoint), lora_alpha=1.0),
            device=device,
        )
    finally:
        infer_core.ReactiveGWMModel = original


def build_grid(output: Path, destinations: list[Path]) -> Path:
    grid = output / "action-grid-3x3.mp4"
    if grid.exists():
        raise FileExistsError(f"Refusing to overwrite grid: {grid}")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for destination in destinations:
        command.extend(["-i", str(destination)])
    labels = [path.stem.replace("-", "_").upper() for path in destinations]
    filters = []
    for index, label in enumerate(labels):
        filters.append(
            f"[{index}:v]scale=416:240,drawtext=text='{label}':"
            "x=8:y=8:fontsize=18:fontcolor=white:"
            f"box=1:boxcolor=black@0.6[v{index}]"
        )
    layout = "0_0|416_0|832_0|0_240|416_240|832_240|0_480|416_480|832_480"
    filters.append(
        "".join(f"[v{index}]" for index in range(len(destinations)))
        + f"xstack=inputs=9:layout={layout}[grid]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[grid]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(grid),
        ]
    )
    subprocess.run(command, check=True)
    return grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--metadata",
        default=str(ISAAC_DATA_ROOT / "metadata_vanilla_strategy_unique.csv"),
    )
    parser.add_argument("--dataset_base", default=str(ISAAC_DATA_ROOT))
    parser.add_argument("--base_model_root", default=str(WAN_ROOT.parents[1]))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--row_index", type=int, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--action_cfg_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--action",
        action="append",
        choices=action_names(),
        help="Repeat to run a subset; default runs no-action plus all 8 actions.",
    )
    parser.add_argument("--skip_grid", action="store_true")
    parser.add_argument(
        "--manifest_name",
        default="eval_manifest.json",
        help="Distinct name for parallel action shards.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    rows = read_rows(args.metadata)
    if not 0 <= args.row_index < len(rows):
        raise IndexError(f"row_index {args.row_index} outside {len(rows)} rows")
    row = rows[args.row_index]
    video_path = Path(args.dataset_base) / row["video"]
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video is missing: {video_path}")

    requested = args.action or list(action_names())
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate actions were requested")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destinations = [
        output / f"{action_names().index(name):02d}-{name.lower()}.mp4"
        for name in requested
    ]
    conflicts = [path for path in destinations if path.exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite inference: {conflicts[0]}")

    input_image = infer_core.first_frame_pil(str(video_path), height=480, width=832)
    pipe = build_isaac_pipe(args.base_model_root, checkpoint, args.device)
    jobs = []
    for index, (name, destination) in enumerate(zip(requested, destinations)):
        action = make_eval_action(name, args.device)
        frames = infer_core.run_inference(
            pipe,
            prompt=args.prompt,
            keyboard_action=action,
            input_image=input_image,
            num_frames=ISAAC_RAW_FRAMES,
            height=480,
            width=832,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            action_cfg_scale=args.action_cfg_scale,
            seed=args.seed,
        )
        save_video(frames, str(destination), fps=20, quality=5)
        jobs.append(
            {
                "index": index,
                "action": name,
                "pressed_frames": [] if name == NO_ACTION else [1, 100],
                "video": str(destination.resolve()),
            }
        )
        print(f"[isaac-eval] {index + 1}/{len(requested)} -> {destination}", flush=True)

    grid = None
    if not args.skip_grid and len(destinations) == 9:
        grid = build_grid(output, destinations)
    report = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": "isaac_v1",
        "checkpoint": str(checkpoint.resolve()),
        "fixed_condition": {
            "metadata": str(Path(args.metadata).resolve()),
            "row_index": args.row_index,
            "source_video": str(video_path.resolve()),
            "prompt": args.prompt,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "action_cfg_scale": args.action_cfg_scale,
            "height": 480,
            "width": 832,
            "num_frames": ISAAC_RAW_FRAMES,
        },
        "jobs": jobs,
        "grid": str(grid.resolve()) if grid else None,
    }
    manifest_path = output / args.manifest_name
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
    atomic_runtime_json(manifest_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
