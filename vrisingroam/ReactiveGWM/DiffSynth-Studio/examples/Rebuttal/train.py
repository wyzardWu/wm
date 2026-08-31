#!/usr/bin/env python3
"""Unified training entry for the three SF2 rebuttal experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import accelerate


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.Rebuttal.checkpoint_io import (  # noqa: E402
    RebuttalCheckpointLogger,
    atomic_write_json,
    build_base_model_fingerprint,
    infer_checkpoint_step,
    resolve_v3_resume,
)
from examples.Rebuttal.preflight import (  # noqa: E402
    validate_cache,
    validate_formal_hyperparameters,
    validate_metadata,
    validate_model_assets,
    visible_cuda_devices,
)
from examples.Rebuttal.runner import launch_rebuttal_training  # noqa: E402
from examples.Rebuttal.isaac_model import (  # noqa: E402
    IsaacReactiveGWMModel,
    audit_isaac_action_embedders,
)
from examples.Rebuttal.isaac_profile import (  # noqa: E402
    ISAAC_PROFILE,
    get_isaac_action_op,
)
from examples.Rebuttal.trainable_policy import (  # noqa: E402
    apply_full_dit_policy,
    apply_hybrid_cross_lora_policy,
    audit_full_dit_policy,
    audit_gradient_contract,
    audit_hybrid_cross_lora_policy,
)
from examples.Rebuttal.variants import (  # noqa: E402
    CROSS_ATTN_LORA_TARGET,
    DATA_ROOT,
    FORMAL_DEFAULTS,
    HERE as REBUTTAL_ROOT,
    LORA_ALPHA,
    LORA_RANK,
    TOKENIZER_PATH,
    WAN_ROOT,
    WAN_MODEL_KWARGS,
    resolve_variant,
    variant_choices,
    wan_model_paths_json,
)


BASE_TRAIN_DIR = REPO_ROOT / "examples/ReactiveGWM/model_training"
BASE_TRAIN_FILE = BASE_TRAIN_DIR / "train.py"


def _load_base_training_module() -> ModuleType:
    """Load the existing trainer under an isolated module name, read-only."""

    reactive_root = BASE_TRAIN_DIR.parent
    for path in (BASE_TRAIN_DIR, reactive_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_name = "_rebuttal_readonly_reactivegwm_train"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, BASE_TRAIN_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base trainer: {BASE_TRAIN_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SF2 rebuttal training: V1 vanilla+strategy, V2 strategy-only, "
            "or V3 cross-attn LoRA plus non-cross full fine-tune."
        )
    )
    parser.add_argument("--variant", required=True, choices=variant_choices())
    parser.add_argument("--dataset_base_path", default=str(DATA_ROOT))
    parser.add_argument(
        "--dataset_metadata_path",
        default=None,
        help="Defaults to the selected variant's authoritative metadata.",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--wan_root", default=str(WAN_ROOT))
    parser.add_argument("--tokenizer_path", default=str(TOKENIZER_PATH))
    parser.add_argument("--use_cached_dataset", action="store_true")
    parser.add_argument("--cache_root", default=None)

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=FORMAL_DEFAULTS["learning_rate"],
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=FORMAL_DEFAULTS["weight_decay"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=FORMAL_DEFAULTS["gradient_accumulation_steps"],
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=FORMAL_DEFAULTS["max_train_steps"],
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=FORMAL_DEFAULTS["save_steps"],
    )
    parser.add_argument(
        "--prompt_dropout_prob",
        type=float,
        default=FORMAL_DEFAULTS["prompt_dropout_prob"],
    )
    parser.add_argument(
        "--action_dropout_prob",
        type=float,
        default=FORMAL_DEFAULTS["action_dropout_prob"],
    )
    parser.add_argument(
        "--dataset_repeat",
        type=int,
        default=FORMAL_DEFAULTS["dataset_repeat"],
    )
    parser.add_argument(
        "--dataset_num_workers",
        type=int,
        default=FORMAL_DEFAULTS["dataset_num_workers"],
    )
    parser.add_argument("--height", type=int, default=FORMAL_DEFAULTS["height"])
    parser.add_argument("--width", type=int, default=FORMAL_DEFAULTS["width"])
    parser.add_argument(
        "--num_frames",
        type=int,
        default=FORMAL_DEFAULTS["num_frames"],
    )
    parser.add_argument(
        "--action_hold_window",
        type=int,
        default=FORMAL_DEFAULTS["action_hold_window"],
    )
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="V1/V2 weight-only step-N.safetensors resume.",
    )
    parser.add_argument(
        "--resume_manifest",
        default=None,
        help="V3 weight-only step-N.manifest.json resume.",
    )
    parser.add_argument(
        "--resume_state",
        default=None,
        help="Optional Accelerate state-N directory; mutually exclusive with weights.",
    )
    parser.add_argument("--save_full_state", action="store_true")
    parser.add_argument("--allow_checkpoint_overwrite", action="store_true")
    parser.add_argument("--allow_existing_output", action="store_true")
    parser.add_argument("--skip_resume_hash_check", action="store_true")

    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help=(
            "Force the 16-row, two-step, non-cached correctness run. "
            "This is the only mode allowed to differ from the formal recipe."
        ),
    )
    parser.add_argument(
        "--cached_smoke_test",
        action="store_true",
        help="Isaac-only four-process, two-step smoke through the formal cache path.",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Validate configuration and inputs without loading the 5B model.",
    )
    parser.add_argument(
        "--expected_num_processes",
        type=int,
        default=FORMAL_DEFAULTS["num_processes"],
    )
    parser.add_argument("--initialize_model_on_cpu", action="store_true")
    parser.add_argument("--find_unused_parameters", action="store_true")
    return parser


def _resolve_mode(args):
    spec = resolve_variant(args.variant)
    if args.smoke_test and args.cached_smoke_test:
        raise ValueError("--smoke_test and --cached_smoke_test are mutually exclusive")
    args.variant = spec.key
    # Fixed arguments consumed by the read-only base dataset/training module.
    args.data_file_keys = "video,action"
    args.max_pixels = 1920 * 1080
    args.use_csv_prompt = True
    args.prompt_column = "prompt"
    if args.smoke_test or args.cached_smoke_test:
        args.max_train_steps = 2
        args.save_steps = 1
        args.dataset_repeat = 1
        args.dataset_num_workers = 0
        args.expected_num_processes = 4 if spec.is_isaac else 1
        if args.cached_smoke_test and not spec.is_isaac:
            raise ValueError("--cached_smoke_test is only supported for isaac_v1")
        if args.smoke_test and (args.use_cached_dataset or args.cache_root):
            raise ValueError("Smoke tests must use the non-cached data path")
        if args.cached_smoke_test:
            if not args.use_cached_dataset or not args.cache_root:
                raise ValueError(
                    "Cached smoke requires --use_cached_dataset and --cache_root"
                )
            if args.dataset_metadata_path is None:
                args.dataset_metadata_path = str(spec.metadata_path)
            elif Path(args.dataset_metadata_path).resolve() != spec.metadata_path.resolve():
                raise ValueError(
                    f"Cached smoke must use authoritative metadata: {spec.metadata_path}"
                )
        elif args.dataset_metadata_path is None:
            args.dataset_metadata_path = str(
                REBUTTAL_ROOT / "generated" / f"metadata_{spec.key}_smoke16.csv"
            )
    else:
        validate_formal_hyperparameters(args)
        if not args.use_cached_dataset or not args.cache_root:
            raise ValueError(
                "Formal training requires --use_cached_dataset and --cache_root"
            )
        if args.dataset_metadata_path is None:
            args.dataset_metadata_path = str(spec.metadata_path)
        elif Path(args.dataset_metadata_path).resolve() != spec.metadata_path.resolve():
            raise ValueError(
                f"Formal {spec.key} training must use its authoritative metadata: "
                f"{spec.metadata_path}"
            )
    return spec


def _validate_scalar_arguments(args) -> None:
    positive = {
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_steps": args.max_train_steps,
        "save_steps": args.save_steps,
        "dataset_repeat": args.dataset_repeat,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "action_hold_window": args.action_hold_window,
        "expected_num_processes": args.expected_num_processes,
    }
    bad = {key: value for key, value in positive.items() if value <= 0}
    if bad:
        raise ValueError(f"Arguments must be positive: {bad}")
    for name in ("prompt_dropout_prob", "action_dropout_prob"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name} must be in [0, 1], got {value}")
    if args.height % 16 or args.width % 16:
        raise ValueError("Height and width must both be divisible by 16")
    if (args.num_frames - 1) % 4:
        raise ValueError("num_frames must satisfy (num_frames - 1) % 4 == 0")


def _resolve_resume(args, spec):
    choices = [
        bool(args.resume_checkpoint),
        bool(args.resume_manifest),
        bool(args.resume_state),
    ]
    if sum(choices) > 1:
        raise ValueError(
            "--resume_checkpoint, --resume_manifest, and --resume_state "
            "are mutually exclusive"
        )
    resume_full = None
    resume_lora = None
    initial_step = 0

    if spec.is_hybrid and args.resume_checkpoint:
        raise ValueError("V3 resumes from --resume_manifest, not --resume_checkpoint")
    if not spec.is_hybrid and args.resume_manifest:
        raise ValueError("V1/V2 resume from --resume_checkpoint, not a V3 manifest")

    if args.resume_checkpoint:
        path = Path(args.resume_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {path}")
        resume_full = str(path)
        initial_step = infer_checkpoint_step(path)
    elif args.resume_manifest:
        full, lora, initial_step, _ = resolve_v3_resume(
            args.resume_manifest,
            verify_hashes=not args.skip_resume_hash_check,
        )
        resume_full = str(full)
        resume_lora = str(lora)
    elif args.resume_state:
        path = Path(args.resume_state)
        if not path.is_dir():
            raise FileNotFoundError(f"Resume state is missing: {path}")
        initial_step = infer_checkpoint_step(path)

    if initial_step >= args.max_train_steps:
        raise ValueError(
            f"Resume step {initial_step} is not below max step "
            f"{args.max_train_steps}"
        )
    return resume_full, resume_lora, initial_step


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _output_preflight(
    args, *, is_main_process: bool, create: bool = True
) -> dict[str, Any]:
    output = Path(args.output_path)
    if is_main_process:
        if output.exists():
            entries = list(output.iterdir()) if output.is_dir() else [output]
            is_resume = bool(
                args.resume_checkpoint or args.resume_manifest or args.resume_state
            )
            if entries and not (args.allow_existing_output or is_resume):
                raise FileExistsError(
                    "Output directory is non-empty; refusing accidental reuse: "
                    f"{output}"
                )
        if create:
            output.mkdir(parents=True, exist_ok=True)
    existing_parent = _nearest_existing_parent(output)
    usage = shutil.disk_usage(existing_parent)
    return {
        "path": str(output.resolve()),
        "filesystem": str(existing_parent.resolve()),
        "free_bytes": usage.free,
        "free_gib": round(usage.free / (1 << 30), 2),
    }


def _resource_preflight(args, *, actual_processes: int | None = None) -> dict[str, Any]:
    visible = visible_cuda_devices()
    if not visible:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must explicitly select the training GPUs"
        )
    if len(visible) != args.expected_num_processes:
        raise RuntimeError(
            f"Visible GPU count {len(visible)} does not match expected "
            f"{args.expected_num_processes}: {visible}"
        )
    if args.variant == "isaac_v1" and visible != ("0", "3", "4", "5"):
        raise RuntimeError(
            "Isaac requires CUDA_VISIBLE_DEVICES=0,3,4,5, "
            f"got {','.join(visible)}"
        )
    if actual_processes is not None and actual_processes != args.expected_num_processes:
        raise RuntimeError(
            f"Accelerate process count {actual_processes} does not match expected "
            f"{args.expected_num_processes}"
        )
    return {
        "cuda_visible_devices": list(visible),
        "expected_num_processes": args.expected_num_processes,
        "actual_processes": actual_processes,
    }


def _public_args(args) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in {"skip_resume_hash_check"}
    }


def _build_model(
    args,
    spec,
    accelerator,
    *,
    resume_full: str | None,
    resume_lora: str | None,
):
    base = _load_base_training_module()
    if base.WAN_MODEL_KWARGS != WAN_MODEL_KWARGS:
        raise RuntimeError(
            "Read-only ReactiveGWM architecture drifted from the locked "
            "Rebuttal WAN_MODEL_KWARGS"
        )
    if spec.is_isaac:
        # The base trainer is loaded under a private module name, so replacing
        # this constructor cannot change shared SF2/SF3 imports.
        base.ReactiveGWMModel = IsaacReactiveGWMModel
        profile = ISAAC_PROFILE
    else:
        profile = base.get_profile("sf2")
    model = base.ReactiveGWMTrainingModule(
        profile=profile,
        model_paths=wan_model_paths_json(Path(args.wan_root)),
        model_id_with_origin_paths=None,
        tokenizer_path=args.tokenizer_path,
        trainable_models=None if spec.is_hybrid else "dit",
        lora_base_model="dit" if spec.is_hybrid else None,
        lora_target_modules=CROSS_ATTN_LORA_TARGET if spec.is_hybrid else "",
        lora_rank=LORA_RANK,
        lora_checkpoint=resume_lora,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs="input_image",
        fp8_models=None,
        offload_models=None,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        trainable_filter="",
        trainable_filter_exclude="",
        use_csv_prompt=True,
        prompt_column="prompt",
        use_cached_dataset=args.use_cached_dataset,
        resume_from_ckpt=resume_full,
        action_dropout_prob=args.action_dropout_prob,
        prompt_dropout_prob=args.prompt_dropout_prob,
    )
    if spec.is_hybrid:
        apply_hybrid_cross_lora_policy(model.pipe.dit)
        audit = audit_hybrid_cross_lora_policy(model.pipe.dit)
    else:
        apply_full_dit_policy(model.pipe.dit)
        audit = audit_full_dit_policy(model.pipe.dit)
        if spec.is_isaac:
            audit_isaac_action_embedders(model.pipe.dit)
    return base, profile, model, audit


def _build_dataset(base, args, spec, profile):
    if not spec.is_isaac:
        return base.build_dataset(args, profile)
    # Both dataset implementations resolve get_action_op from module globals.
    # Patch only for this Isaac process; shared source files remain untouched.
    base.get_action_op = get_isaac_action_op
    if args.use_cached_dataset:
        import cached_dataset

        cached_dataset.get_action_op = get_isaac_action_op
    return base.build_dataset(args, profile)


def _run_config(
    *,
    args,
    spec,
    metadata,
    cache,
    resources,
    output,
    model_assets,
    base_fingerprint,
    parameter_audit=None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": spec.key,
        "prompt_mode": spec.prompt_mode,
        "parameter_mode": spec.parameter_mode,
        "cold_start": "raw Wan2.2-TI2V-5B",
        "action_input": True,
        "lora": (
            {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "target_modules": CROSS_ATTN_LORA_TARGET,
                "norm3_frozen": True,
            }
            if spec.is_hybrid
            else None
        ),
        "arguments": _public_args(args),
        "metadata": metadata.to_dict(),
        "cache": cache.to_dict() if cache else None,
        "resources": resources,
        "output": output,
        "model_assets": model_assets,
        "base_model": base_fingerprint,
        "parameter_audit": parameter_audit,
    }


def _gradient_auditor(args, spec, accelerator):
    results: dict[str, Any] = {}
    target_steps = {1, 2} if spec.is_hybrid else {1}

    def audit(training_model, step: int) -> None:
        if step not in target_steps:
            return
        result = audit_gradient_contract(
            training_model.pipe.dit,
            mode=spec.parameter_mode,
            step=step,
            require_lora_a=spec.is_hybrid and step >= 2,
        )
        if accelerator.is_main_process:
            results[str(step)] = result.to_dict()
            atomic_write_json(
                Path(args.output_path) / "gradient_audit.json",
                {
                    "variant": spec.key,
                    "required_steps": sorted(target_steps),
                    "results": results,
                },
            )
            print(
                f"[gradient-audit] step={step} "
                f"{json.dumps(result.representatives, sort_keys=True)}",
                flush=True,
            )

    return audit


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = _resolve_mode(args)
    _validate_scalar_arguments(args)
    resume_full, resume_lora, initial_step = _resolve_resume(args, spec)

    metadata = validate_metadata(
        spec,
        args.dataset_metadata_path,
        allow_isaac_subset=bool(args.smoke_test and spec.is_isaac),
    )
    model_assets = validate_model_assets(args.wan_root, args.tokenizer_path)
    cache = None
    if args.use_cached_dataset:
        cache = validate_cache(
            args.cache_root,
            metadata,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            variant=spec.key,
        )
    base_fingerprint = build_base_model_fingerprint(args.wan_root)

    if args.preflight_only:
        resources = _resource_preflight(args)
        output = _output_preflight(args, is_main_process=True, create=False)
        config = _run_config(
            args=args,
            spec=spec,
            metadata=metadata,
            cache=cache,
            resources=resources,
            output=output,
            model_assets=model_assets,
            base_fingerprint=base_fingerprint,
        )
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )
    resources = _resource_preflight(args, actual_processes=accelerator.num_processes)
    output = _output_preflight(args, is_main_process=accelerator.is_main_process)
    accelerator.wait_for_everyone()
    accelerate.utils.set_seed(args.seed, device_specific=True)

    base, profile, model, parameter_audit = _build_model(
        args,
        spec,
        accelerator,
        resume_full=resume_full,
        resume_lora=resume_lora,
    )
    dataset = _build_dataset(base, args, spec, profile)
    config = _run_config(
        args=args,
        spec=spec,
        metadata=metadata,
        cache=cache,
        resources=resources,
        output=output,
        model_assets=model_assets,
        base_fingerprint=base_fingerprint,
        parameter_audit=parameter_audit.to_dict(),
    )
    if accelerator.is_main_process:
        config_name = (
            f"resume-step-{initial_step}.json" if initial_step else "run_config.json"
        )
        atomic_write_json(Path(args.output_path) / config_name, config)
        print(parameter_audit.to_json(), flush=True)
    accelerator.wait_for_everyone()

    logger = RebuttalCheckpointLogger(
        args.output_path,
        variant=spec.key,
        base_model_fingerprint=base_fingerprint,
        metadata=metadata.to_dict(),
        training_config=_public_args(args),
        parameter_audit=parameter_audit.to_dict(),
        initial_step=initial_step,
        allow_overwrite=args.allow_checkpoint_overwrite,
    )
    result = launch_rebuttal_training(
        accelerator,
        dataset,
        model,
        logger,
        args=args,
        gradient_auditor=_gradient_auditor(args, spec, accelerator),
    )
    if accelerator.is_main_process:
        print(
            f"[complete] variant={spec.key} "
            f"steps={result.initial_step}->{result.final_step}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
