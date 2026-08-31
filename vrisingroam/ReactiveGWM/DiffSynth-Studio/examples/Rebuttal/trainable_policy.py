"""Trainable-parameter policies and fail-fast audits for rebuttal training."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .variants import CROSS_ATTN_LORA_TARGET, LORA_ALPHA, LORA_RANK


_NORM3_RE = re.compile(r"(?:^|\.)blocks\.\d+\.norm3(?:\.|$)")
_TARGET_LORA_RE = re.compile(
    r"(?:^|\.)blocks\.\d+\.cross_attn\.(q|k|v|o)" r"\.lora_([AB])(?:\.|$)"
)


def is_lora_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def is_cross_attention_parameter(name: str) -> bool:
    return ".cross_attn." in f".{name}"


def is_norm3_parameter(name: str) -> bool:
    return _NORM3_RE.search(name) is not None


def is_target_cross_lora_parameter(name: str) -> bool:
    return _TARGET_LORA_RE.search(name) is not None


def apply_full_dit_policy(dit: torch.nn.Module) -> None:
    """Full-fine-tune every DiT parameter."""

    dit.train()
    dit.requires_grad_(True)


def apply_hybrid_cross_lora_policy(dit: torch.nn.Module) -> None:
    """Freeze the original cross branch and full-fine-tune everything else.

    PEFT LoRA parameters under cross-attention q/k/v/o stay trainable.
    Original q/k/v/o parameters, cross-attention internal norms, and the
    enclosing block's norm3 are frozen.
    """

    dit.train()
    for name, parameter in dit.named_parameters():
        if is_cross_attention_parameter(name):
            parameter.requires_grad = is_target_cross_lora_parameter(name)
        elif is_norm3_parameter(name):
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True


@dataclass(frozen=True)
class ParameterCount:
    tensors: int
    scalars: int


@dataclass(frozen=True)
class TrainableAudit:
    mode: str
    total: ParameterCount
    trainable: ParameterCount
    frozen: ParameterCount
    trainable_by_category: dict[str, ParameterCount]
    lora_modules: int
    lora_tensors: int
    lora_scalars: int
    first_trainable_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class GradientAudit:
    mode: str
    step: int
    representatives: dict[str, dict[str, float | str]]
    frozen_tensors_checked: int

    def to_dict(self) -> dict:
        return asdict(self)


def _count(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]]
) -> ParameterCount:
    params = list(named_parameters)
    return ParameterCount(
        tensors=len(params),
        scalars=sum(parameter.numel() for _, parameter in params),
    )


def _category(name: str) -> str:
    if "action_embedders" in name:
        return "action"
    if ".self_attn." in f".{name}":
        return "self_attn"
    if is_cross_attention_parameter(name) or is_norm3_parameter(name):
        return "cross_branch"
    if ".ffn." in f".{name}":
        return "ffn"
    return "other"


def _build_audit(dit: torch.nn.Module, mode: str) -> TrainableAudit:
    named = list(dit.named_parameters())
    trainable = [(name, param) for name, param in named if param.requires_grad]
    frozen = [(name, param) for name, param in named if not param.requires_grad]
    lora = [(name, param) for name, param in named if is_lora_parameter(name)]
    lora_modules = {name.split(".lora_", maxsplit=1)[0] for name, _ in lora}
    categories = {}
    for category in ("action", "self_attn", "cross_branch", "ffn", "other"):
        categories[category] = _count(
            (name, param) for name, param in trainable if _category(name) == category
        )
    return TrainableAudit(
        mode=mode,
        total=_count(named),
        trainable=_count(trainable),
        frozen=_count(frozen),
        trainable_by_category=categories,
        lora_modules=len(lora_modules),
        lora_tensors=len(lora),
        lora_scalars=sum(param.numel() for _, param in lora),
        first_trainable_names=tuple(name for name, _ in trainable[:24]),
    )


def _assert_component_is_trainable(
    named: list[tuple[str, torch.nn.Parameter]], marker: str
) -> None:
    selected = [(name, param) for name, param in named if marker in name]
    if not selected:
        raise AssertionError(f"Required parameter component is absent: {marker}")
    bad = [name for name, param in selected if not param.requires_grad]
    if bad:
        raise AssertionError(
            f"{marker} contains frozen parameters; first entries: {bad[:8]}"
        )


def audit_full_dit_policy(dit: torch.nn.Module) -> TrainableAudit:
    """Assert that the entire DiT, including action and cross-attn, is trainable."""

    named = list(dit.named_parameters())
    if not named:
        raise AssertionError("DiT has no parameters")
    lora_names = [name for name, _ in named if is_lora_parameter(name)]
    if lora_names:
        raise AssertionError(
            f"Full-DiT mode unexpectedly contains LoRA: {lora_names[:8]}"
        )
    frozen = [name for name, param in named if not param.requires_grad]
    if frozen:
        raise AssertionError(
            f"Full-DiT mode has frozen parameters; first entries: {frozen[:8]}"
        )
    for marker in ("action_embedders", "self_attn", "cross_attn", "ffn"):
        _assert_component_is_trainable(named, marker)
    return _build_audit(dit, mode="full_dit")


def audit_hybrid_cross_lora_policy(
    dit: torch.nn.Module,
    *,
    expected_lora_modules: int = 120,
    expected_lora_tensors: int = 240,
    expected_lora_scalars: int = 23_592_960,
) -> TrainableAudit:
    """Assert the exact V3 frozen/trainable contract."""

    named = list(dit.named_parameters())
    if not named:
        raise AssertionError("DiT has no parameters")

    errors: list[str] = []
    for name, parameter in named:
        if is_cross_attention_parameter(name):
            expected = is_target_cross_lora_parameter(name)
        elif is_norm3_parameter(name):
            expected = False
        else:
            expected = True
        if parameter.requires_grad != expected:
            errors.append(
                f"{name}: requires_grad={parameter.requires_grad}, expected={expected}"
            )
    if errors:
        raise AssertionError(
            "Hybrid policy mismatch; first entries:\n  " + "\n  ".join(errors[:12])
        )

    lora_names = [name for name, _ in named if is_lora_parameter(name)]
    wrong_lora = [
        name for name in lora_names if not is_target_cross_lora_parameter(name)
    ]
    if wrong_lora:
        raise AssertionError(
            f"LoRA escaped cross-attn q/k/v/o; first entries: {wrong_lora[:8]}"
        )

    for marker in ("action_embedders", "self_attn", "ffn"):
        _assert_component_is_trainable(named, marker)
    if not any(is_cross_attention_parameter(name) for name, _ in named):
        raise AssertionError("No cross-attention parameters found")
    if not any(is_norm3_parameter(name) for name, _ in named):
        raise AssertionError("No blocks.*.norm3 parameters found")

    audit = _build_audit(dit, mode="hybrid_cross_lora")
    expected = {
        "LoRA modules": (audit.lora_modules, expected_lora_modules),
        "LoRA tensors": (audit.lora_tensors, expected_lora_tensors),
        "LoRA scalars": (audit.lora_scalars, expected_lora_scalars),
    }
    mismatches = [
        f"{label}: got {actual:,}, expected {wanted:,}"
        for label, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise AssertionError("Hybrid LoRA count mismatch: " + "; ".join(mismatches))
    return audit


def build_and_audit_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Create the one uniform AdamW group and prove exact parameter coverage."""

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise AssertionError("No trainable parameters are available for AdamW")
    if len({id(parameter) for parameter in trainable}) != len(trainable):
        raise AssertionError("The model exposes duplicate trainable parameter objects")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    audit_optimizer(
        optimizer,
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return optimizer


def audit_optimizer(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> None:
    """Assert one uniform group, no duplicates, omissions, or frozen params."""

    if not isinstance(optimizer, torch.optim.AdamW):
        raise AssertionError(f"Expected AdamW, got {type(optimizer).__name__}")
    if len(optimizer.param_groups) != 1:
        raise AssertionError(
            f"Expected exactly one optimizer group, got {len(optimizer.param_groups)}"
        )
    group = optimizer.param_groups[0]
    if group["lr"] != learning_rate or group["weight_decay"] != weight_decay:
        raise AssertionError(
            "Optimizer hyperparameters differ from the uniform contract: "
            f"lr={group['lr']}, weight_decay={group['weight_decay']}"
        )

    optimizer_params = list(group["params"])
    optimizer_ids = [id(parameter) for parameter in optimizer_params]
    if len(set(optimizer_ids)) != len(optimizer_ids):
        raise AssertionError("A trainable parameter appears more than once in AdamW")

    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    actual = set(optimizer_ids)
    if actual != expected:
        raise AssertionError(
            "AdamW parameter coverage mismatch: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    frozen_ids = {
        id(parameter) for parameter in model.parameters() if not parameter.requires_grad
    }
    if actual & frozen_ids:
        raise AssertionError("AdamW contains frozen parameters")


def _gradient_representative(
    named: list[tuple[str, torch.nn.Parameter]],
    *,
    label: str,
) -> dict[str, float | str]:
    with_grad = [
        (name, parameter)
        for name, parameter in named
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not with_grad:
        raise AssertionError(f"No gradient reached required component: {label}")
    # Small tensors minimize audit overhead while still proving the path is live.
    for name, parameter in sorted(with_grad, key=lambda item: item[1].numel()):
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"Non-finite gradient in {label}: {name}")
        norm = float(gradient.float().norm().item())
        if norm > 0:
            return {"name": name, "norm": norm}
    raise AssertionError(f"Every observed gradient is zero in component: {label}")


def audit_gradient_contract(
    dit: torch.nn.Module,
    *,
    mode: str,
    step: int,
    require_lora_a: bool = False,
) -> GradientAudit:
    """Probe representative gradients and prove frozen branches stayed untouched."""

    named = list(dit.named_parameters())
    frozen_with_grad = [
        name
        for name, parameter in named
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_grad:
        raise AssertionError(
            "Frozen parameters unexpectedly received gradients: "
            f"{frozen_with_grad[:8]}"
        )

    selectors = {
        "action": lambda name: "action_embedders" in name,
        "self_attn": lambda name: ".self_attn." in f".{name}",
        "ffn": lambda name: ".ffn." in f".{name}",
    }
    if mode == "full_dit":
        selectors["cross_attn"] = is_cross_attention_parameter
    elif mode == "hybrid_cross_lora":
        selectors["lora_B"] = lambda name: ".lora_B." in name
        if require_lora_a:
            selectors["lora_A"] = lambda name: ".lora_A." in name
    else:
        raise ValueError(f"Unknown gradient-audit mode: {mode}")

    representatives = {
        label: _gradient_representative(
            [(name, parameter) for name, parameter in named if selector(name)],
            label=label,
        )
        for label, selector in selectors.items()
    }
    return GradientAudit(
        mode=mode,
        step=step,
        representatives=representatives,
        frozen_tensors_checked=sum(
            1 for _, parameter in named if not parameter.requires_grad
        ),
    )


__all__ = [
    "CROSS_ATTN_LORA_TARGET",
    "LORA_ALPHA",
    "LORA_RANK",
    "TrainableAudit",
    "GradientAudit",
    "apply_full_dit_policy",
    "apply_hybrid_cross_lora_policy",
    "audit_full_dit_policy",
    "audit_gradient_contract",
    "audit_hybrid_cross_lora_policy",
    "audit_optimizer",
    "build_and_audit_optimizer",
]
