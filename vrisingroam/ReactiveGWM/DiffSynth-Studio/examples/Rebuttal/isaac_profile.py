"""Isaac/Gurdy Jr data contract isolated from the shared SF2/SF3 profiles."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffsynth.core.data.operators import RouteByType
from examples.ReactiveGWM.data.profiles import GameProfile


ISAAC_ACTION_COLUMNS = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "SHOOT_UP",
    "SHOOT_DOWN",
    "SHOOT_LEFT",
    "SHOOT_RIGHT",
)
ISAAC_RAW_FRAMES = 101
ISAAC_LATENT_FRAMES = 26
ISAAC_ACTION_INDICES = (0, *range(1, ISAAC_RAW_FRAMES, 4))

ISAAC_PROFILE = GameProfile(
    name="isaac",
    description="The Binding of Isaac: Isaac vs Gurdy Jr",
    button_cols=ISAAC_ACTION_COLUMNS,
    fixed_prompt="The Binding of Isaac gameplay",
    action_presets=(),
    default_height=480,
    default_width=832,
    default_num_frames=ISAAC_RAW_FRAMES,
    default_use_csv_prompt=True,
    default_action_hold_window=1,
)


def read_isaac_action(path: str | Path) -> torch.Tensor:
    """Read one parquet under the strict 101x8 binary Isaac contract."""

    path = Path(path)
    frame = pd.read_parquet(path)
    missing = [column for column in ISAAC_ACTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Isaac action parquet lacks columns {missing}: {path}")
    if len(frame) != ISAAC_RAW_FRAMES:
        raise ValueError(
            f"Isaac action parquet must have exactly {ISAAC_RAW_FRAMES} rows, "
            f"got {len(frame)}: {path}"
        )
    selected = frame.loc[:, list(ISAAC_ACTION_COLUMNS)]
    if selected.isna().any().any():
        raise ValueError(f"Isaac action parquet contains NaN: {path}")
    values = selected.to_numpy()
    if not np.isin(values, (0, 1, False, True)).all():
        unique = np.unique(values)
        raise ValueError(
            f"Isaac action values must be binary 0/1, got {unique[:16]!r}: {path}"
        )
    return torch.from_numpy(values.astype(np.float32, copy=False))


def get_isaac_action_op(
    profile: GameProfile,
    base_path: str,
    num_frames: int,
    hold_window: int | None = None,
) -> RouteByType:
    """Build the strict Isaac parquet operator with no truncation or upsampling."""

    if profile.name != ISAAC_PROFILE.name:
        raise ValueError(f"Isaac action operator received profile {profile.name!r}")
    if num_frames != ISAAC_RAW_FRAMES:
        raise ValueError(
            f"Isaac num_frames must be {ISAAC_RAW_FRAMES}, got {num_frames}"
        )
    window = profile.default_action_hold_window if hold_window is None else hold_window
    if window != 1:
        raise ValueError(f"Isaac action_hold_window must be 1, got {window}")

    def process_parquet(rel_path: str) -> torch.Tensor:
        action = read_isaac_action(os.path.join(base_path, rel_path))
        return action.unsqueeze(0)

    return RouteByType(operator_map=[(str, process_parquet)])


__all__ = [
    "ISAAC_ACTION_COLUMNS",
    "ISAAC_ACTION_INDICES",
    "ISAAC_LATENT_FRAMES",
    "ISAAC_PROFILE",
    "ISAAC_RAW_FRAMES",
    "get_isaac_action_op",
    "read_isaac_action",
]
