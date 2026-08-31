"""Isaac-only ReactiveGWM DiT with strict causal-first-4n1 action alignment."""

from __future__ import annotations

import torch

from diffsynth.models.reactive_gwm_dit import ReactiveGWMModel
from examples.Rebuttal.isaac_profile import (
    ISAAC_ACTION_COLUMNS,
    ISAAC_ACTION_INDICES,
    ISAAC_LATENT_FRAMES,
    ISAAC_RAW_FRAMES,
)


class IsaacReactiveGWMModel(ReactiveGWMModel):
    """Keep the shared architecture but replace adaptive pooling for Isaac."""

    def __init__(self, num_buttons: int = len(ISAAC_ACTION_COLUMNS), **kwargs):
        if num_buttons != len(ISAAC_ACTION_COLUMNS):
            raise ValueError(
                f"Isaac requires {len(ISAAC_ACTION_COLUMNS)} buttons, "
                f"got {num_buttons}"
            )
        super().__init__(num_buttons=num_buttons, **kwargs)

    def prepare_action_binned(
        self, keyboard_action: torch.Tensor | None, f: int
    ) -> torch.Tensor | None:
        if keyboard_action is None:
            return None
        if keyboard_action.ndim != 3:
            raise ValueError(
                "Isaac keyboard_action must have shape [B, 101, 8], "
                f"got {tuple(keyboard_action.shape)}"
            )
        _, raw_steps, buttons = keyboard_action.shape
        if raw_steps != ISAAC_RAW_FRAMES or buttons != len(ISAAC_ACTION_COLUMNS):
            raise ValueError(
                "Isaac keyboard_action must have shape [B, 101, 8], "
                f"got {tuple(keyboard_action.shape)}"
            )
        if f != ISAAC_LATENT_FRAMES:
            raise ValueError(
                f"Isaac latent action steps must be {ISAAC_LATENT_FRAMES}, got {f}"
            )
        indices = torch.as_tensor(
            ISAAC_ACTION_INDICES,
            device=keyboard_action.device,
            dtype=torch.long,
        )
        result = keyboard_action.index_select(1, indices)
        expected = (keyboard_action.shape[0], ISAAC_LATENT_FRAMES, buttons)
        if tuple(result.shape) != expected:
            raise RuntimeError(
                f"Isaac causal_first_4n1 produced {tuple(result.shape)}, "
                f"expected {expected}"
            )
        return result


def audit_isaac_action_embedders(model: IsaacReactiveGWMModel) -> dict[str, int]:
    embedders = model.action_embedders
    if len(embedders) != 30:
        raise ValueError(f"Isaac requires 30 action embedders, got {len(embedders)}")
    for index, embedder in enumerate(embedders):
        if embedder.in_features != 8 or embedder.out_features != 3072:
            raise ValueError(
                f"action_embedders.{index} has shape "
                f"{embedder.in_features}->{embedder.out_features}"
            )
        if embedder.bias is not None:
            raise ValueError(f"action_embedders.{index} unexpectedly has bias")
    parameters = sum(parameter.numel() for module in embedders for parameter in module.parameters())
    if parameters != 737_280:
        raise ValueError(
            f"Isaac action embedder parameter count must be 737280, got {parameters}"
        )
    return {"count": len(embedders), "parameters": parameters}


__all__ = ["IsaacReactiveGWMModel", "audit_isaac_action_embedders"]
