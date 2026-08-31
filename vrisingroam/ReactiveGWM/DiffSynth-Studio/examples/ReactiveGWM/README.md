# ReactiveGWM — Reactive Game-World Model

Action-conditioned video diffusion training and inference on top of the
**Wan2.2-TI2V-5B** backbone, with per-block independent action embedders
(see `diffsynth.models.reactive_gwm_dit.ReactiveGWMModel`). Currently ships
two registered game profiles:

| profile | game | NPC | resolution | use_csv_prompt default |
|---|---|---|---|---|
| `sf2` | Street Fighter II (Genesis) | Guile @ Air Force Base | 480×608 | true |
| `sf3` | Street Fighter III (CPS3) | Ibuki @ Ibuki's Stage | 480×832 | false |

Both share the same 10-button schema (UP/DOWN/LEFT/RIGHT + Y/X/Z/A/B/C) but the
13-action `eval_action.py` grid differs because SF2 and SF3 map light/medium/
heavy punch+kick to different physical buttons (see `data/profiles.py` for the
exact mapping or each game's source-of-truth dataset README).

This module is a **self-contained additional feature** of DiffSynth-Studio.
It only adds two library files and one example tree; nothing in upstream
`diffsynth/` is modified.

---

## 1. Layout

```
examples/ReactiveGWM/
├── data/                                    Generic, profile-driven utilities
│   ├── profiles.py                          GameProfile dataclass + PROFILES registry
│   ├── action_utils.py                      hold_last_upsample, get_action_op(profile, ...)
│   └── prompt_utils.py                      resolve_prompt(row, profile, ...)
├── model_training/
│   ├── train.py                             ReactiveGWMTrainingModule, --game flag
│   └── cached_dataset.py                    CachedReactiveGWMDataset (profile-driven)
├── inference/
│   ├── infer_core.py                        build_pipe / run_inference (game-agnostic)
│   ├── _mp_runner.py                        Multi-GPU job runner (uses profile.button_cols)
│   ├── metrics.py                           Region-based motion energy & MSE
│   └── eval_action.py                       Profile-driven 13-action axis eval
├── scripts/
│   └── precompute_cache.py                  Unified VAE+T5 cache precompute, --game flag
├── sf2/                                     SF2-specific shell launchers + extras
│   ├── launch/{training,eval,cache}/*.sh
│   └── extras/                              eval_strategy.py (7 v5 strategies),
│                                            eval_focus.py, eval_sf2all_strategy.py,
│                                            prefill_v4_from_v2.py / prefill_v5_from_v4.py,
│                                            rebuild_cache_fixed_prompt.py
└── sf3/                                     SF3-specific shell launchers + extras
    ├── launch/{training,eval,cache}/*.sh
    └── extras/                              eval_strategy.py (9 v5cat strategies),
                                             eval_strategy_6{,_sf2prompt,_sf2prompt_v2}.py,
                                             infer_sf3.py, infer_strategy_fixedinput.py,
                                             select_strategy_samples.py,
                                             precompute_visual_t5_cache.py,
                                             rebuild_cache_fixed_prompt.py,
                                             _make_strategy6_report.py / _swap_xattn.py /
                                             _tidy_strategy6_outputs.py
```

The two library files this example depends on (added to upstream `diffsynth/`):

```
diffsynth/models/reactive_gwm_dit.py     ReactiveGWMModel = WanModel + per-block action embedders
diffsynth/pipelines/reactive_gwm.py      ReactiveGWMPipeline + WanVideoUnit_* + model_fn_wan_video
                                         (supports keyboard_action and action_cfg_scale)
```

---

## 2. Quickstart — SF2

```bash
# 1. (Optional) Build VAE+T5 cache to speed training ~30%:
bash examples/ReactiveGWM/sf2/launch/cache/run_cache_4shard.sh

# 2. Full DiT fine-tune (8 GPUs, ~2.05 s/step uncached):
bash examples/ReactiveGWM/sf2/launch/training/full_5s_480x608.sh

# 3. Action-axis evaluation (13 button presets):
bash examples/ReactiveGWM/sf2/launch/eval/eval_action.sh <run_name> <step>
```

Or invoke directly:

```bash
python examples/ReactiveGWM/inference/eval_action.py \
  --game sf2 \
  --full_ckpt /.../step-N.safetensors \
  --output_dir /.../action_eval --reference_clip /.../clip.mp4
```

---

## 3. Quickstart — SF3

```bash
# 1. Build cache (8 shards parallel):
bash examples/ReactiveGWM/sf3/launch/cache/run_cache_8shard_v5cat.sh

# 2. Cached training (480×832):
bash examples/ReactiveGWM/sf3/launch/training/attempt_sf3_5s_480x832_cached.sh

# 3. Strategy-axis evaluation (9 v5cat strategies × 3 action presets):
bash examples/ReactiveGWM/sf3/launch/eval/eval_strategy.sh <run_name> <step>
```

---

## 4. Adding a new game

1. Append a `GameProfile(...)` to `data/profiles.py` and register it under `PROFILES`.
2. Build a cache:
   `python examples/ReactiveGWM/scripts/precompute_cache.py --game <name> ...`.
3. Train: `python examples/ReactiveGWM/model_training/train.py --game <name> ...`.
4. Optional: add a `<name>/launch/` and `<name>/extras/` subfolder to host
   shell wrappers and game-specific scripts.

The framework imposes nothing about the dataset beyond:
- a metadata CSV with at least `video` and `action` columns,
- a parquet action file at the path each `action` cell points to, with one
  column per name in `profile.button_cols`.

---

## 5. Training modes

All driven by `model_training/train.py`'s flags; combine freely:

| Goal | Flags |
|---|---|
| Full DiT fine-tune | `--trainable_models dit` |
| Scoped (keep only cross_attn) | `--trainable_models dit --trainable_filter cross_attn` |
| Scoped (drop cross_attn, train rest) | `--trainable_models dit --trainable_filter_exclude cross_attn` |
| LoRA on cross_attn + self_attn (rank=32) | `--lora_base_model dit --lora_target_modules cross_attn,self_attn --lora_rank 32` |
| + cache acceleration | `--use_cached_dataset --cache_root <path>` |
| + CFG-style prompt training | `--prompt_dropout_prob 0.1` |
| + CFG-style action training | `--action_dropout_prob 0.1` |
| Resume from saved DiT state dict (no opt state) | `--resume_from_ckpt <step-N.safetensors>` |

Cache mode automatically drops the four pipeline units replaced by cached
tensors (`PromptEmbedder` / `InputVideoEmbedder` / `ImageEmbedderFused` /
`NoiseInitializer`), drops `pipe.vae` / `pipe.text_encoder`, and reproduces the
NoiseInitializer's CUDA-randn so RNG parity with the non-cached path is exact.

---

## 6. Profile-default vs CLI override

Most launch scripts elide arguments that match a profile's defaults
(`default_height`, `default_width`, `default_num_frames`, `default_use_csv_prompt`).
Pass them explicitly to override. `--use_csv_prompt true|false` accepts an
explicit value to force either mode regardless of the profile.

---

## 7. Library code (added to upstream)

The two files added to `diffsynth/` are self-contained — they do not require
or depend on any other Training_Wan-era patches that aren't already in
upstream DiffSynth-Studio:

- `diffsynth/models/reactive_gwm_dit.py` (~500 lines):
  redefines its own `WanModel`/`DiTBlock`/`RMSNorm`/etc. and adds
  `ReactiveGWMModel` which adds per-block `nn.Linear(num_buttons, dim)`
  action embedders. Only imports `SimpleAdapter` and
  `gradient_checkpoint_forward` from upstream.

- `diffsynth/pipelines/reactive_gwm.py` (~1340 lines):
  `ReactiveGWMPipeline = BasePipeline` with full set of
  `WanVideoUnit_*` units and its own `model_fn_wan_video` that supports
  `keyboard_action` and `action_cfg_scale`. Imports `WanModel` /
  `sinusoidal_embedding_1d` from `diffsynth.models.wan_video_dit` (upstream).

Importing them looks like:

```python
from diffsynth.models.reactive_gwm_dit import ReactiveGWMModel
from diffsynth.pipelines.reactive_gwm import ReactiveGWMPipeline, ModelConfig
```

Nothing else in `diffsynth/` is touched — every existing example
(wanvideo, qwen_image, flux2, ace_step, etc.) keeps working unchanged.
