#!/usr/bin/env python3
"""Fixed-condition strategy-axis inference for rebuttal checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REACTIVE_ROOT = REPO_ROOT / "examples/ReactiveGWM"
for path in (REPO_ROOT, REACTIVE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.profiles import get_profile  # noqa: E402

from diffsynth.utils.data import save_video  # noqa: E402

from examples.ReactiveGWM.inference.infer_core import (  # noqa: E402
    CkptSpec,
    build_pipe,
    first_frame_pil,
    load_action_from_parquet,
    run_inference,
)
from examples.Rebuttal.atomic_runtime import atomic_runtime_json  # noqa: E402
from examples.Rebuttal.checkpoint_io import resolve_v3_resume  # noqa: E402
from examples.Rebuttal.prepare_metadata import extract_strategy  # noqa: E402
from examples.Rebuttal.variants import (  # noqa: E402
    DATA_ROOT,
    STRUCTURED_METADATA,
    WAN_ROOT,
    resolve_variant,
    variant_choices,
)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def strategy_catalog(path: str | Path = STRUCTURED_METADATA) -> list[str]:
    strategies = {extract_strategy(row["prompt"]) for row in read_rows(path)}
    return sorted(strategies)


def replace_strategy(prompt: str, strategy: str, prompt_mode: str) -> str:
    strategy = extract_strategy(strategy)
    old = extract_strategy(prompt)
    if prompt_mode == "strategy_only":
        return strategy
    if prompt_mode == "vanilla_strategy":
        return prompt[: -len(old)].rstrip() + " " + strategy
    if prompt_mode == "structured":
        return prompt[: -len(old)] + strategy
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def resolve_checkpoint(args, spec) -> CkptSpec:
    if bool(args.checkpoint) == bool(args.manifest):
        raise ValueError("Provide exactly one of --checkpoint or --manifest")
    if args.manifest:
        if not spec.is_hybrid:
            raise ValueError("--manifest is only valid for V3 split checkpoints")
        full, lora, _, _ = resolve_v3_resume(
            args.manifest,
            verify_hashes=not args.skip_checkpoint_hash_check,
        )
        return CkptSpec(
            full_ckpt=str(full),
            lora_ckpt=str(lora),
            lora_alpha=1.0,
        )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")
    return CkptSpec(full_ckpt=str(checkpoint), lora_alpha=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=variant_choices())
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--row_index", type=int, default=0)
    parser.add_argument("--dataset_base", default=str(DATA_ROOT))
    parser.add_argument("--base_model_root", default=str(WAN_ROOT.parents[1]))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        help="Full Strategy(...); repeat for a subset. Default: all nine.",
    )
    parser.add_argument("--max_strategies", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--num_frames", type=int, default=101)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--action_cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_checkpoint_hash_check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = resolve_variant(args.variant)
    metadata_path = Path(args.metadata or spec.metadata_path)
    rows = read_rows(metadata_path)
    if not 0 <= args.row_index < len(rows):
        raise IndexError(
            f"row_index {args.row_index} outside metadata with {len(rows)} rows"
        )
    row = rows[args.row_index]
    strategies = args.strategy or strategy_catalog()
    strategies = [extract_strategy(strategy) for strategy in strategies]
    if args.max_strategies > 0:
        strategies = strategies[: args.max_strategies]
    if len(set(strategies)) != len(strategies):
        raise ValueError("Duplicate strategies were requested")

    checkpoint = resolve_checkpoint(args, spec)
    profile = get_profile("sf2")
    pipe = build_pipe(
        profile,
        args.base_model_root,
        checkpoint,
        device=args.device,
    )
    dataset_base = Path(args.dataset_base)
    video_path = dataset_base / row["video"]
    action_path = dataset_base / row["action"]
    input_image = first_frame_pil(
        str(video_path),
        height=args.height,
        width=args.width,
    )
    action = load_action_from_parquet(
        profile,
        str(action_path),
        args.num_frames,
        device=args.device,
        hold_window=10,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, strategy in enumerate(strategies):
        prompt = replace_strategy(row["prompt"], strategy, spec.prompt_mode)
        destination = output / f"strategy-{index:02d}.mp4"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite inference: {destination}")
        frames = run_inference(
            pipe,
            prompt=prompt,
            keyboard_action=action,
            input_image=input_image,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            action_cfg_scale=args.action_cfg_scale,
            seed=args.seed,
        )
        save_video(frames, str(destination), fps=20, quality=5)
        jobs.append(
            {
                "index": index,
                "strategy": strategy,
                "prompt": prompt,
                "video": str(destination.resolve()),
            }
        )
        print(f"[eval] {index + 1}/{len(strategies)} -> {destination}", flush=True)

    report = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": spec.key,
        "checkpoint": {
            "full": checkpoint.full_ckpt,
            "lora": checkpoint.lora_ckpt,
            "lora_alpha": checkpoint.lora_alpha,
        },
        "fixed_condition": {
            "metadata": str(metadata_path.resolve()),
            "row_index": args.row_index,
            "video": str(video_path.resolve()),
            "action": str(action_path.resolve()),
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "action_cfg_scale": args.action_cfg_scale,
        },
        "jobs": jobs,
    }
    atomic_runtime_json(output / "eval_manifest.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
