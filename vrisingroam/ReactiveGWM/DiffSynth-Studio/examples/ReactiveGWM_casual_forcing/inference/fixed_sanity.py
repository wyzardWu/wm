"""Fixed sanity prompt + action loader.

从 dataset clip `clip_idx` (默认 0) 读 prompt + parquet action, hold-last 上采样后
按 `target_pixel_frames` 截断/idle 填充, 同时记录视频文件路径 (caller VAE encode 首帧).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent.parent
_REACTIVE_GWM = _HERE.parent / "ReactiveGWM"
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from data.action_utils import hold_last_upsample  # noqa: E402
from data.profiles import get_profile  # noqa: E402
from data.prompt_utils import resolve_prompt  # noqa: E402


def _resolve_action_buttons(profile, action_name: str) -> tuple[str, ...]:
    """Map an action alias to the profile-specific button tuple."""
    name = action_name.strip().lower().replace("-", "_")
    if name in {"idle", "noop", "none"}:
        return ()
    if name in {"jump", "up"}:
        return ("UP",)

    preset_alias = {
        "light_punch": "lightP",
        "lightp": "lightP",
        "lp": "lightP",
        "heavy_kick": "heavyK",
        "heavyk": "heavyK",
        "hk": "heavyK",
    }.get(name)
    if preset_alias is not None:
        for slug, buttons in profile.action_presets:
            if preset_alias in slug:
                return buttons
        raise ValueError(f"no {preset_alias} preset found for game {profile.name}")

    raw_button = action_name.strip().upper()
    if raw_button in profile.button_cols:
        return (raw_button,)
    raise ValueError(f"unknown action: {action_name!r}")


def _resolve_fill_buttons(profile, fill_action: str) -> tuple[str, ...]:
    """Map a fill-action name to its button-name tuple."""
    return _resolve_action_buttons(profile, fill_action)


def _build_repeating_action_sequence(
    profile,
    action_sequence: str,
    target_pixel_frames: int,
    segment_frames: int | None,
) -> torch.Tensor:
    """Build [T, K] by repeating comma-separated action aliases."""
    actions = [x.strip() for x in action_sequence.split(",") if x.strip()]
    if not actions:
        raise ValueError("action_sequence must contain at least one action")
    seg = int(segment_frames or profile.default_action_hold_window)
    if seg <= 0:
        raise ValueError(f"action_segment_frames must be positive, got {seg}")

    action = torch.zeros(target_pixel_frames, profile.num_buttons, dtype=torch.float32)
    button_groups = [_resolve_action_buttons(profile, name) for name in actions]
    for start in range(0, target_pixel_frames, seg):
        buttons = button_groups[(start // seg) % len(button_groups)]
        end = min(start + seg, target_pixel_frames)
        for name in buttons:
            action[start:end, profile.button_index(name)] = 1.0
    return action


def load_fixed_sanity(
    *,
    metadata_path: str,
    dataset_base: str,
    game: str = "sf3",
    clip_idx: int = 0,
    target_pixel_frames: int = 1500,
    use_csv_prompt: bool = True,
    prompt_column: str = "prompt",
    fill_action: str = "idle",
    action_sequence: str | None = None,
    action_segment_frames: int | None = None,
) -> dict[str, Any]:
    """Return `{prompt, action [1, T, K], video_path}`.

    Pads the action sequence (beyond the clip's real length) up to
    `target_pixel_frames`. `fill_action='idle'` pads with zeros; other values
    (e.g. 'light_punch') hold a fixed button down for the padded tail.
    """
    profile = get_profile(game)
    df = pd.read_csv(metadata_path)
    row = df.iloc[clip_idx].to_dict()
    prompt = resolve_prompt(row, profile, use_csv_prompt, prompt_column)

    if action_sequence is not None:
        action = _build_repeating_action_sequence(
            profile, action_sequence, target_pixel_frames, action_segment_frames
        )
    else:
        # Action parquet
        action_path = os.path.join(dataset_base, row["action"])
        adf = pd.read_parquet(action_path)
        cols = list(profile.button_cols)
        arr = adf[cols].values.astype("float32")
        action = torch.tensor(arr)
        action = hold_last_upsample(action, window=profile.default_action_hold_window)

        # Pad / truncate to target length
        T, K = action.shape
        if T < target_pixel_frames:
            pad = torch.zeros(target_pixel_frames - T, K, dtype=action.dtype)
            fill_buttons = _resolve_fill_buttons(profile, fill_action)
            for name in fill_buttons:
                pad[:, profile.button_index(name)] = 1.0
            action = torch.cat([action, pad], dim=0)
        elif T > target_pixel_frames:
            action = action[:target_pixel_frames]

    # Video path (for first-frame anchor encode)
    video_path = os.path.join(dataset_base, row["video"])

    return {
        "prompt": prompt,
        "action": action.unsqueeze(0),  # [1, T, K]
        "video_path": video_path,
    }
