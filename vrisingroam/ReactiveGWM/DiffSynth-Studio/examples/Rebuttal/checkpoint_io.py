"""Checkpoint contracts for the three rebuttal training variants."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file

from .trainable_policy import (
    is_cross_attention_parameter,
    is_lora_parameter,
    is_norm3_parameter,
    is_target_cross_lora_parameter,
)
from .variants import CROSS_ATTN_LORA_TARGET, LORA_ALPHA, LORA_RANK


CHECKPOINT_SCHEMA_VERSION = 1
DIT_PREFIX = "pipe.dit."
_STEP_RE = re.compile(r"(?:^|[./_-])step[-_]?(\d+)(?:[./_-]|$)")


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def save_safetensors_atomic(
    state_dict: Mapping[str, torch.Tensor], path: str | Path
) -> None:
    """Atomically save a small/local state dict (used by tests and export)."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.safetensors"
    )
    serializable = {
        key: tensor.detach().cpu().contiguous() for key, tensor in state_dict.items()
    }
    save_file(serializable, temporary)
    os.replace(temporary, destination)


def _accelerator_save_atomic(
    accelerator,
    state_dict: Mapping[str, torch.Tensor],
    path: Path,
) -> None:
    """Use Accelerate's sharding-aware saver, then atomically publish the file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.safetensors"
    )
    accelerator.save(state_dict, temporary, safe_serialization=True)
    os.replace(temporary, path)


def checkpoint_state_stats(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    dtypes: dict[str, int] = {}
    scalars = 0
    bytes_total = 0
    for tensor in state_dict.values():
        count = tensor.numel()
        scalars += count
        bytes_total += count * tensor.element_size()
        name = str(tensor.dtype).removeprefix("torch.")
        dtypes[name] = dtypes.get(name, 0) + count
    return {
        "keys": len(state_dict),
        "scalars": scalars,
        "tensor_bytes": bytes_total,
        "dtypes_by_scalar": dtypes,
    }


def split_hybrid_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    full = {}
    lora = {}
    for name, tensor in state_dict.items():
        (lora if is_lora_parameter(name) else full)[name] = tensor
    validate_hybrid_checkpoint_parts(full, lora)
    return full, lora


def _lora_side_and_prefix(name: str) -> tuple[str, str]:
    for side in ("A", "B"):
        for marker in (f".lora_{side}.default.weight", f".lora_{side}.weight"):
            if name.endswith(marker):
                return side, name[: -len(marker)]
    raise ValueError(f"Unsupported LoRA checkpoint key: {name}")


def lora_pairs(
    lora_state: Mapping[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora_state.items():
        side, prefix = _lora_side_and_prefix(name)
        if side in pairs.setdefault(prefix, {}):
            raise ValueError(f"Duplicate LoRA {side} tensor for {prefix}")
        pairs[prefix][side] = tensor
    incomplete = {
        prefix: sorted({"A", "B"} - set(sides))
        for prefix, sides in pairs.items()
        if set(sides) != {"A", "B"}
    }
    if incomplete:
        raise ValueError(f"Incomplete LoRA pairs: {incomplete}")
    return {prefix: (sides["A"], sides["B"]) for prefix, sides in pairs.items()}


def validate_full_dit_checkpoint(state_dict: Mapping[str, torch.Tensor]) -> None:
    if not state_dict:
        raise ValueError("Full-DiT checkpoint is empty")
    lora = [name for name in state_dict if is_lora_parameter(name)]
    if lora:
        raise ValueError(f"Full-DiT checkpoint contains LoRA keys: {lora[:8]}")
    for marker in ("action_embedders", "self_attn", "cross_attn", "ffn"):
        if not any(marker in name for name in state_dict):
            raise ValueError(f"Full-DiT checkpoint lacks required component: {marker}")


def validate_hybrid_checkpoint_parts(
    full_state: Mapping[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
) -> None:
    if not full_state:
        raise ValueError("V3 full delta is empty")
    if not lora_state:
        raise ValueError("V3 LoRA checkpoint is empty")

    illegal_full = [
        name
        for name in full_state
        if is_cross_attention_parameter(name)
        or is_norm3_parameter(name)
        or is_lora_parameter(name)
    ]
    if illegal_full:
        raise ValueError(
            "V3 full delta contains frozen cross-branch keys: " f"{illegal_full[:8]}"
        )
    for marker in ("action_embedders", "self_attn", "ffn"):
        if not any(marker in name for name in full_state):
            raise ValueError(f"V3 full delta lacks required component: {marker}")

    illegal_lora = [
        name
        for name in lora_state
        if not is_lora_parameter(name) or not is_target_cross_lora_parameter(name)
    ]
    if illegal_lora:
        raise ValueError(
            "V3 LoRA checkpoint contains non-cross-q/k/v/o keys: " f"{illegal_lora[:8]}"
        )
    lora_pairs(lora_state)


def infer_checkpoint_step(path: str | Path) -> int:
    match = _STEP_RE.search(str(path))
    if match is None:
        raise ValueError(f"Cannot infer training step from checkpoint path: {path}")
    return int(match.group(1))


def validate_state_shapes(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    model_state = model.state_dict()
    missing = [name for name in state_dict if name not in model_state]
    bad_shapes = [
        (name, tuple(tensor.shape), tuple(model_state[name].shape))
        for name, tensor in state_dict.items()
        if name in model_state and tensor.shape != model_state[name].shape
    ]
    if missing or bad_shapes:
        raise ValueError(
            "Checkpoint does not fit the model: "
            f"unknown_keys={missing[:8]}, shape_mismatches={bad_shapes[:8]}"
        )


def fuse_hybrid_checkpoint_into_model(
    model: torch.nn.Module,
    *,
    full_state: Mapping[str, torch.Tensor],
    lora_state: Mapping[str, torch.Tensor],
    rank: int = LORA_RANK,
    alpha: int = LORA_ALPHA,
) -> int:
    """Overlay the full delta and fuse LoRA into an unwrapped raw DiT."""

    validate_hybrid_checkpoint_parts(full_state, lora_state)
    validate_state_shapes(model, full_state)
    result = model.load_state_dict(full_state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected full-delta keys: {result.unexpected_keys[:8]}")

    modules = dict(model.named_modules())
    pairs = lora_pairs(lora_state)
    scale = alpha / rank
    for prefix, (lora_a, lora_b) in pairs.items():
        module = modules.get(prefix)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"LoRA target is not a raw Linear module: {prefix}")
        if lora_a.shape[0] != rank or lora_b.shape[1] != rank:
            raise ValueError(
                f"LoRA rank mismatch at {prefix}: "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}, rank={rank}"
            )
        # Match GeneralLoRALoader exactly: cast A/B to the pipeline dtype before
        # matrix multiplication. This matters for bit-level BF16 equivalence
        # between split loading and the exported merged checkpoint.
        delta = torch.mm(
            lora_b.to(device=module.weight.device, dtype=module.weight.dtype),
            lora_a.to(device=module.weight.device, dtype=module.weight.dtype),
        )
        if delta.shape != module.weight.shape:
            raise ValueError(
                f"Fused LoRA shape mismatch at {prefix}: "
                f"delta={tuple(delta.shape)}, weight={tuple(module.weight.shape)}"
            )
        module.weight.data.add_(
            (delta * scale).to(device=module.weight.device, dtype=module.weight.dtype)
        )
    return len(pairs)


def build_base_model_fingerprint(wan_root: str | Path) -> dict[str, Any]:
    """Record stable, cheap base identity without re-hashing ~30 GB each save."""

    root = Path(wan_root).resolve()
    files = []
    for path in sorted(root.glob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        prefix_digest = hashlib.sha256()
        with path.open("rb") as handle:
            prefix_digest.update(handle.read(1 << 20))
        files.append(
            {
                "name": path.name,
                "bytes": stat.st_size,
                "sha256_first_1m": prefix_digest.hexdigest(),
            }
        )
    return {"root": str(root), "files": files}


def load_and_validate_hybrid_manifest(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema: {manifest.get('schema_version')}"
        )
    if manifest.get("variant") != "v3":
        raise ValueError(f"Expected V3 manifest, got {manifest.get('variant')!r}")
    lora_config = manifest.get("lora") or {}
    expected_lora = {
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "target_modules": CROSS_ATTN_LORA_TARGET,
    }
    if any(lora_config.get(key) != value for key, value in expected_lora.items()):
        raise ValueError(
            f"Manifest LoRA config mismatch: got={lora_config}, "
            f"expected={expected_lora}"
        )
    for kind in ("full", "lora"):
        entry = (manifest.get("files") or {}).get(kind) or {}
        file_path = path.parent / entry.get("name", "")
        if not file_path.is_file():
            raise FileNotFoundError(f"Manifest {kind} file is missing: {file_path}")
        if verify_hashes and sha256_file(file_path) != entry.get("sha256"):
            raise ValueError(f"Manifest SHA256 mismatch for {file_path}")
    return manifest


class RebuttalCheckpointLogger:
    """Accelerate-compatible logger implementing the locked checkpoint layout."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        variant: str,
        base_model_fingerprint: Mapping[str, Any],
        metadata: Mapping[str, Any],
        training_config: Mapping[str, Any],
        parameter_audit: Mapping[str, Any],
        initial_step: int = 0,
        allow_overwrite: bool = False,
    ):
        if variant not in {"v1", "v2", "v3", "isaac_v1"}:
            raise ValueError(f"Unsupported checkpoint variant: {variant}")
        self.output_path = str(Path(output_path))
        self.variant = variant
        self.base_model_fingerprint = dict(base_model_fingerprint)
        self.metadata = dict(metadata)
        self.training_config = dict(training_config)
        self.parameter_audit = dict(parameter_audit)
        self.num_steps = initial_step
        self.allow_overwrite = allow_overwrite

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs) -> None:
        self.num_steps += 1
        loss = kwargs.get("loss")
        if loss is not None and accelerator.is_main_process:
            value = float(loss.detach().float().item())
            print(f"[step {self.num_steps}] loss={value:.6f}", flush=True)
        if save_steps and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, self.num_steps)

    def on_training_end(self, accelerator, model, save_steps=None) -> None:
        if not save_steps or self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, self.num_steps)

    def _assert_new(self, paths: list[Path]) -> None:
        existing = [str(path) for path in paths if path.exists()]
        if existing and not self.allow_overwrite:
            raise FileExistsError(
                "Refusing to overwrite checkpoint files: " + ", ".join(existing)
            )

    def save_model(self, accelerator, model, step: int) -> None:
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if not accelerator.is_main_process:
            return

        unwrapped = accelerator.unwrap_model(model)
        state_dict = unwrapped.export_trainable_state_dict(
            state_dict, remove_prefix=DIT_PREFIX
        )
        output = Path(self.output_path)
        output.mkdir(parents=True, exist_ok=True)

        if self.variant in {"v1", "v2", "isaac_v1"}:
            validate_full_dit_checkpoint(state_dict)
            checkpoint = output / f"step-{step}.safetensors"
            self._assert_new([checkpoint])
            _accelerator_save_atomic(accelerator, state_dict, checkpoint)
            print(f"[checkpoint] wrote {checkpoint}", flush=True)
            return

        full_state, lora_state = split_hybrid_state_dict(state_dict)
        full_path = output / f"step-{step}.full.safetensors"
        lora_path = output / f"step-{step}.lora.safetensors"
        manifest_path = output / f"step-{step}.manifest.json"
        self._assert_new([full_path, lora_path, manifest_path])
        _accelerator_save_atomic(accelerator, full_state, full_path)
        _accelerator_save_atomic(accelerator, lora_state, lora_path)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "variant": "v3",
            "step": step,
            "base_model": self.base_model_fingerprint,
            "metadata": self.metadata,
            "training_config": self.training_config,
            "parameter_audit": self.parameter_audit,
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "target_modules": CROSS_ATTN_LORA_TARGET,
            },
            "files": {
                "full": {
                    "name": full_path.name,
                    "sha256": sha256_file(full_path),
                    "bytes": full_path.stat().st_size,
                    **checkpoint_state_stats(full_state),
                },
                "lora": {
                    "name": lora_path.name,
                    "sha256": sha256_file(lora_path),
                    "bytes": lora_path.stat().st_size,
                    **checkpoint_state_stats(lora_state),
                },
            },
        }
        atomic_write_json(manifest_path, manifest)
        print(
            f"[checkpoint] wrote {full_path}, {lora_path}, and {manifest_path}",
            flush=True,
        )


def resolve_v3_resume(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> tuple[Path, Path, int, dict[str, Any]]:
    path = Path(manifest_path)
    manifest = load_and_validate_hybrid_manifest(path, verify_hashes=verify_hashes)
    return (
        path.parent / manifest["files"]["full"]["name"],
        path.parent / manifest["files"]["lora"]["name"],
        int(manifest["step"]),
        manifest,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DIT_PREFIX",
    "RebuttalCheckpointLogger",
    "atomic_write_json",
    "build_base_model_fingerprint",
    "checkpoint_state_stats",
    "fuse_hybrid_checkpoint_into_model",
    "infer_checkpoint_step",
    "load_and_validate_hybrid_manifest",
    "resolve_v3_resume",
    "save_safetensors_atomic",
    "sha256_file",
    "split_hybrid_state_dict",
    "validate_full_dit_checkpoint",
    "validate_hybrid_checkpoint_parts",
    "validate_state_shapes",
]
