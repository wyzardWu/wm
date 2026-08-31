"""Shared constants: button order, default prompts/resolutions, negative prompt."""

# Same for SF2 and SF3 — the parquet must have these 10 columns.
SF_BUTTON_COLS = ["UP", "DOWN", "LEFT", "RIGHT", "Y", "X", "Z", "A", "B", "C"]

# Negative prompt copied verbatim from the training-time inference scripts.
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# Per-variant defaults: prompt + native training resolution + action schema.
# button_cols/hold_window MUST mirror the training GameProfile exactly.
VARIANT_DEFAULTS = {
    "sf2": {
        "prompt": "Street Fighter 2 arcade fighting game gameplay",
        "height": 480,
        "width": 608,
        "button_cols": SF_BUTTON_COLS,
        "hold_window": 10,
    },
    "sf3": {
        "prompt": "SF3 Game.",
        "height": 480,
        "width": 832,
        "button_cols": SF_BUTTON_COLS,
        "hold_window": 10,
    },
    "vrising": {
        # Must match training/data/profiles.py VRISING.fixed_prompt verbatim.
        "prompt": (
            "V Rising gameplay, top-down isometric view, a vampire character "
            "roaming the Dunley Farmlands countryside with farms, roads, fences "
            "and villagers."
        ),
        "height": 480,
        "width": 832,
        "button_cols": ["W", "A", "S", "D", "MOUSE0"],
        "hold_window": 1,  # parquets are true per-frame signals
    },
    "eybx": {
        # Must match training/data/profiles.py EYBX.fixed_prompt verbatim.
        "prompt": (
            "No Rest for the Wicked gameplay, top-down isometric view, a knight "
            "roaming through the world."
        ),
        "height": 480,
        "width": 832,
        # SCENE rides as the 6th channel (int id 0..2); consumed only in
        # table mode together with load_scene_context_table().
        "button_cols": ["W", "A", "S", "D", "MOUSE0", "SCENE"],
        "hold_window": 1,
    },
}
