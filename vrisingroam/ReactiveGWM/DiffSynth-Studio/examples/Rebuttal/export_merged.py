#!/usr/bin/env python3
"""Export raw Wan + V3 full delta + fused LoRA as one standard full DiT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REACTIVE_ROOT = REPO_ROOT / "examples/ReactiveGWM"
for path in (REPO_ROOT, REACTIVE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.profiles import get_profile  # noqa: E402

from diffsynth.models.reactive_gwm_dit import ReactiveGWMModel  # noqa: E402
from diffsynth.pipelines.reactive_gwm import (  # noqa: E402
    ModelConfig,
    ReactiveGWMPipeline,
)

from examples.Rebuttal.checkpoint_io import (  # noqa: E402
    atomic_write_json,
    fuse_hybrid_checkpoint_into_model,
    load_and_validate_hybrid_manifest,
    save_safetensors_atomic,
    sha256_file,
    validate_full_dit_checkpoint,
)
from examples.Rebuttal.variants import (  # noqa: E402
    WAN_MODEL_KWARGS,
    WAN_ROOT,
    wan_model_paths_json,
)


def transfer_wan_weights(
    custom_model: torch.nn.Module, pretrained_dit: torch.nn.Module
) -> tuple[int, int]:
    pretrained = pretrained_dit.state_dict()
    custom = custom_model.state_dict()
    compatible = {
        name: tensor
        for name, tensor in pretrained.items()
        if name in custom and tensor.shape == custom[name].shape
    }
    result = custom_model.load_state_dict(compatible, strict=False)
    return len(compatible), len(result.missing_keys)


def build_raw_reactive_dit(
    wan_root: str | Path,
    *,
    device: str,
) -> ReactiveGWMModel:
    paths = json.loads(wan_model_paths_json(Path(wan_root)))
    pipe = ReactiveGWMPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=paths[0])],
        tokenizer_config=None,
    )
    if pipe.dit is None:
        raise RuntimeError("Wan2.2 DiT could not be loaded")
    profile = get_profile("sf2")
    custom = ReactiveGWMModel(
        num_buttons=profile.num_buttons,
        **WAN_MODEL_KWARGS,
    ).to(dtype=torch.bfloat16, device=device)
    transferred, missing = transfer_wan_weights(custom, pipe.dit)
    print(
        f"[cold-start] transferred={transferred}, init-only={missing}",
        flush=True,
    )
    pipe.dit = None
    del pipe
    torch.cuda.empty_cache()
    return custom


def verify_merged_structure(
    path: str | Path,
    expected_state: dict[str, torch.Tensor],
) -> dict:
    expected_keys = set(expected_state)
    shape_mismatches = []
    dtype_mismatches = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        for name in sorted(expected_keys & actual_keys):
            expected = expected_state[name]
            actual_shape = tuple(handle.get_slice(name).get_shape())
            if actual_shape != tuple(expected.shape):
                shape_mismatches.append((name, actual_shape, tuple(expected.shape)))
            actual_dtype = str(handle.get_slice(name).get_dtype()).lower()
            expected_dtype = str(expected.dtype).removeprefix("torch.").lower()
            aliases = {"bf16": "bfloat16", "f16": "float16", "f32": "float32"}
            actual_dtype = aliases.get(actual_dtype, actual_dtype)
            if actual_dtype != expected_dtype:
                dtype_mismatches.append((name, actual_dtype, expected_dtype))
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected or shape_mismatches or dtype_mismatches:
        raise ValueError(
            "Merged checkpoint structure mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shapes={shape_mismatches[:8]}, dtypes={dtype_mismatches[:8]}"
        )
    return {
        "keys": len(expected_keys),
        "scalars": sum(tensor.numel() for tensor in expected_state.values()),
        "dtypes": sorted({str(tensor.dtype) for tensor in expected_state.values()}),
    }


def export_merged(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    wan_root: str | Path = WAN_ROOT,
    device: str = "cuda",
    verify_source_hashes: bool = True,
) -> dict:
    manifest_file = Path(manifest_path).resolve()
    manifest = load_and_validate_hybrid_manifest(
        manifest_file,
        verify_hashes=verify_source_hashes,
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite merged checkpoint: {destination}")

    full_path = manifest_file.parent / manifest["files"]["full"]["name"]
    lora_path = manifest_file.parent / manifest["files"]["lora"]["name"]
    dit = build_raw_reactive_dit(wan_root, device=device)
    full_state = load_file(full_path, device="cpu")
    lora_state = load_file(lora_path, device="cpu")
    fused_modules = fuse_hybrid_checkpoint_into_model(
        dit,
        full_state=full_state,
        lora_state=lora_state,
        rank=manifest["lora"]["rank"],
        alpha=manifest["lora"]["alpha"],
    )
    del full_state, lora_state

    dit = dit.to("cpu")
    torch.cuda.empty_cache()
    state = dit.state_dict()
    validate_full_dit_checkpoint(state)
    save_safetensors_atomic(state, destination)
    structure = verify_merged_structure(destination, state)
    result = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_manifest": str(manifest_file),
        "source_manifest_sha256": sha256_file(manifest_file),
        "base_model": manifest["base_model"],
        "fused_lora_modules": fused_modules,
        "output": {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            **structure,
        },
    }
    atomic_write_json(destination.with_suffix(".manifest.json"), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wan_root", default=str(WAN_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_source_hash_check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_merged(
        manifest_path=args.manifest,
        output_path=args.output,
        wan_root=args.wan_root,
        device=args.device,
        verify_source_hashes=not args.skip_source_hash_check,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
