"""Locked experiment definitions for SF2 rebuttal and isolated Isaac training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

DATA_ROOT = Path("/home/zeqingwang/zeqingwang/ReactiveGWM/ReactiveGWM-Datasets/SF2")
ISAAC_DATA_ROOT = Path(
    "/home/zeqingwang/zeqingwang/datasets/Isaac_processed/final_v1"
)
WAN_ROOT = Path("/nfs/zeqingwang/models/base_model/Wan-AI/Wan2.2-TI2V-5B")
TOKENIZER_PATH = Path(
    "/nfs/zeqingwang/models/base_model/Wan-AI/" "Wan2.1-T2V-1.3B/google/umt5-xxl"
)

STRUCTURED_METADATA = DATA_ROOT / "metadata.csv"
V1_METADATA = HERE / "generated/metadata_v1_vanilla_strategy.csv"
V2_METADATA = HERE / "generated/metadata_v2_strategy_only.csv"
METADATA_MANIFEST = HERE / "generated/metadata_manifest.json"
ISAAC_METADATA = ISAAC_DATA_ROOT / "metadata_vanilla_strategy.csv"

CROSS_ATTN_LORA_TARGET = r".*\.cross_attn\.(q|k|v|o)"
LORA_RANK = 32
LORA_ALPHA = 32

WAN_MODEL_KWARGS = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


@dataclass(frozen=True)
class VariantSpec:
    key: str
    aliases: tuple[str, ...]
    description: str
    prompt_mode: str
    metadata_path: Path
    parameter_mode: str

    @property
    def is_hybrid(self) -> bool:
        return self.parameter_mode == "hybrid_cross_lora"

    @property
    def is_isaac(self) -> bool:
        return self.key == "isaac_v1"


VARIANTS: dict[str, VariantSpec] = {
    "v1": VariantSpec(
        key="v1",
        aliases=("v1", "vanilla_strategy"),
        description="vanilla narration plus one space plus Strategy(...)",
        prompt_mode="vanilla_strategy",
        metadata_path=V1_METADATA,
        parameter_mode="full_dit",
    ),
    "v2": VariantSpec(
        key="v2",
        aliases=("v2", "strategy_only"),
        description="Strategy(...) only",
        prompt_mode="strategy_only",
        metadata_path=V2_METADATA,
        parameter_mode="full_dit",
    ),
    "v3": VariantSpec(
        key="v3",
        aliases=("v3", "hybrid_cross_lora"),
        description="original structured prompt",
        prompt_mode="structured",
        metadata_path=STRUCTURED_METADATA,
        parameter_mode="hybrid_cross_lora",
    ),
    "isaac_v1": VariantSpec(
        key="isaac_v1",
        aliases=("isaac_v1", "isaac_vanilla_strategy"),
        description="Isaac vanilla narration plus natural-language strategy",
        prompt_mode="isaac_vanilla_strategy",
        metadata_path=ISAAC_METADATA,
        parameter_mode="full_dit",
    ),
}

_ALIAS_TO_KEY = {
    alias: spec.key for spec in VARIANTS.values() for alias in spec.aliases
}


def variant_choices() -> tuple[str, ...]:
    """Return every accepted CLI spelling in stable order."""

    return tuple(_ALIAS_TO_KEY)


def resolve_variant(name: str) -> VariantSpec:
    """Resolve a short or descriptive variant name."""

    try:
        return VARIANTS[_ALIAS_TO_KEY[name]]
    except KeyError as exc:
        choices = ", ".join(variant_choices())
        raise ValueError(f"Unknown variant {name!r}; choose one of: {choices}") from exc


def wan_model_paths_json(wan_root: Path = WAN_ROOT) -> str:
    """Build the exact JSON payload expected by DiffSynth model loading."""

    paths: list[object] = [
        [
            str(wan_root / f"diffusion_pytorch_model-{part:05d}-of-00003.safetensors")
            for part in range(1, 4)
        ],
        str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth"),
        str(wan_root / "Wan2.2_VAE.pth"),
    ]
    return json.dumps(paths, separators=(",", ":"))


FORMAL_DEFAULTS = {
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "num_processes": 6,
    "max_train_steps": 30_000,
    "save_steps": 1_000,
    "prompt_dropout_prob": 0.1,
    "action_dropout_prob": 0.0,
    "dataset_repeat": 1,
    "dataset_num_workers": 4,
    "height": 480,
    "width": 608,
    "num_frames": 101,
    "action_hold_window": 10,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
}

# V2 was explicitly approved for the four currently available GPUs. Keep the
# original six-process contract for V1 and V3 so their recipes do not drift.
FORMAL_NUM_PROCESSES = {
    "v1": 6,
    "v2": 4,
    "v3": 6,
    "isaac_v1": 4,
}

ISAAC_FORMAL_DEFAULTS = {
    **FORMAL_DEFAULTS,
    "num_processes": 4,
    "width": 832,
    "action_hold_window": 1,
}


def formal_defaults_for(variant: str) -> dict[str, object]:
    return ISAAC_FORMAL_DEFAULTS if variant == "isaac_v1" else FORMAL_DEFAULTS
