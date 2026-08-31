# Changelog

## 1.3.0 - 2026-08-25

### Added

- DFR can finish 4K in a tiled spatial epilogue instead of denoising the full canvas in stage 2. `--spatial-upscalings 2` keeps stage 2 at half resolution and upscales in that epilogue, which is the path for better-quality 4K (use `3840x2176`; `3840x2160` is not on the size grid). Default `--spatial-upscalings 1` is unchanged.
- Added `ltx_pipelines.dfr_mgpu`, a multi-GPU DFR runner with the same flags as `ltx_pipelines.dfr_pipeline`.
- Added keyframe-aware diffusion-VAE decoding, which sharpens detail at the frames a generator anchored on. Pass keyframe latents into `VideoDecoder.decode_video(..., keyframes=)`. Needs a keyframe-trained video VAE; older checkpoints decode as before.
- `--compile max_video_tokens=N max_audio_tokens=N` (and the matching `CompilationConfig` fields) cap CUDA-graph input memory when `capture=true`, so several captured shapes share one buffer sized for the largest instead of paying for each separately.
- `DiffusionStage.with_model_wrapper()` runs a callable on each built transformer before denoising, and still composes with block streaming and quantization.
- `VideoDecoder.decode_single_frames()` decodes independent one-frame latents without neighbouring frames bleeding into each other.
- `DecodeKeyframes.for_frame_span()` and `.crop_spatial()` crop keyframe planes to a decode window, keeping the nearest anchors outside the window so a tiled decode matches a whole one.
- `DiffusionStage.__call__` accepts optional `audio_fps=` when audio should be sized from a different frame rate than the video RoPE time base.

### Changed

- `DFRPipeline` and `ltx_pipelines.dfr_mgpu` take a distilled checkpoint (`--distilled-checkpoint-path`) instead of a full checkpoint plus a distilled LoRA. Detailing IC-LoRA and optional user LoRAs are unchanged.
- DFR temporal rounds use `--temporal-upscalings` instead of `--temporal-upsample-rounds`.
- DFR snaps transformer conditioning fps to 60 whenever playback fps is above 30 (previously only values above 60 were capped), so 48 and 50 fps stay in the trained range. Playback fps is unchanged.
- `--diffvae-optimization combined_compile` no longer requires the `natten` extra. Without it the mode falls back to Triton, or to PyTorch when Triton is unavailable.
- Automatic decode tiling sizes tiles for keyframe-aware decoding as well as plain decoding, so a keyframe decode is less likely to run out of memory on a clip a plain decode fits.
- Video pipelines now return a named `PipelineOutput` instead of a 3- or 4-tuple. Extra fields are `keyframes` and `video_latent`. Existing unpacking must switch to `result.video`, `result.audio`, and so on.
- `VideoDecoder.decode_video()` takes `keyframes=` on every implementation. The diffusion decoder uses them; conv logs a warning and decodes without them; distributed splits them across ranks. Callers that never pass keyframes are unaffected; decoder implementations must accept the keyword.
- NVFP4 quantization and CuTe DSL DiffVAE kernels (`blackwell_dsl`) now support Jetson Thor. Build NVFP4 with the arch-specific `110a` target.

### Removed

- `DiffusionVideoDecoder.decode_video_with_keyframes`; use `decode_video(..., keyframes=)`.
- `DFROutput`; use `PipelineOutput`.
- `--checkpoint-path` and `--distilled-lora` on `ltx_pipelines.dfr_pipeline` and `ltx_pipelines.dfr_mgpu`; pass `--distilled-checkpoint-path` instead.
- `--num-generated-keyframes` on DFR; slot positions come from DFR's segment grid.
- `--temporal-upsample-rounds` on DFR; use `--temporal-upscalings`.

### Fixed

- Automatic decode tiling no longer OOMs in colour conversion on clips that looked like they had spare VRAM. The budget now reserves encoder memory for each chunk, so long videos decode in temporal pieces instead of as one oversize frame dump.
- A pipelines-only install (`ltx-core` / `ltx-pipelines`, no trainer) no longer fails to load Gemma 4 for missing torchvision.
- `uv sync` on macOS no longer fails looking for CUDA-only torch wheels. Macs use PyPI (MPS on Apple Silicon); Linux and Windows still use the CUDA 13.2 indexes.
- `--quantization fp8-cast` with a LoRA no longer crashes on pre-Ada GPUs such as RTX 3060 or A100.
- Blockwise FP8/FP6 quantization no longer fails on very long sequences or token counts that are not a multiple of 4.
- Multi-GPU DFR no longer crashes at tiling resolve, during carry-keyframe decode, or by dropping context at a temporal tile boundary.
- DFR image-to-video stills land on the correct frame after temporal upsampling (opening-frame `frame_idx=0` was already correct).
- `AUTO_TILING` is recognised after it crosses a process boundary, so multi-GPU pipelines resolve automatic tiling instead of rejecting the sentinel.

## 1.2.0 - 2026-08-11

Support for LTX 2.5

### Added

- Added support for newer LTX checkpoints, including Gemma 4 text encoders, checkpoint-driven architecture selection, compatibility checks between checkpoints and Gemma roots, and LTX 2.5 training workflows.
- Added diffusion-based video VAE decoding with single- and multi-GPU support, optional NATTEN acceleration, and `chunked_eager`, `chunked_compile`, `combined_compile`, and datacenter-Blackwell DSL optimization modes.
- Added caption-based automatic duration prediction for distilled, text-to-audio, and text/image-to-video pipelines. Use `--auto-duration MIN_SECONDS MAX_SECONDS`, or omit `--num-frames` with a compatible checkpoint.
- Added checkpoint-aware `--vae-checkpoint-path` overrides and `--diffvae-optimization` pipeline options.
- Added optional dedicated prompt-enhancement Gemma models through `--prompt-enhancer-gemma-root`, plus `--enhance-static-cache` for reusable enhancement KV caches.
- Added self-managed CUDA graph capture for compiled transformers with `capture=true`, alongside controls for dynamic sequence dimensions and perturbed-block recompilation.
- Added Euler ancestral diffusion sampling.
- Added checkpoint-aware size-, count-, and automatic-tiling APIs that support non-default VAE compression factors.
- Added reusable model-shell caching, configurable independently from checkpoint-weight caching, through `ModelRegistry`.
- Added disposable model support so cached module structure and non-persistent buffers can survive weight offloading.
- Added fused CuTe DSL diffusion-VAE kernels for datacenter Blackwell GPUs.
- Added model-version-aware image-conditioning compression defaults and expanded checkpoint metadata utilities.
- Added NVFP4 quantization, which cuts transformer memory use and speeds up inference on Blackwell GPUs. `--quantization nvfp4-cast` quantizes a BF16 checkpoint while loading it, and `--quantization nvfp4-prequant` loads an already-quantized NVFP4 checkpoint. Requires a Blackwell GPU and the `ltx-kernels` package.
- Added `DFRPipeline`, which raises output quality and holds detail together noticeably better on fast motion. It generates interior keyframes, adds a full-resolution detailing pass, and can run up to two temporal upsampling rounds for smoother movement. Adds `--detailing-lora`, `--temporal-upsampler-path`, `--temporal-upsample-rounds`, and `--num-generated-keyframes`.
- Added split-checkpoint loading, so you can download only the components a pipeline needs and mix them freely, such as a quantized transformer with a BF16 VAE. Each component has its own flag: `--transformer-path`, `--text-encoder-path`, `--video-vae-path`, `--audio-vae-path`, and `--duration-head-path`. Single-file checkpoints keep working through `--checkpoint-path` and `--distilled-checkpoint-path`.
- Added HDR conditioning and output, so generated video can go into a colour-grading or VFX pipeline without a lossy intermediate. `--hdr` selects the colour space for EXR conditioning and HDR encoding, which writes scene-linear EXR frames alongside a BT.2020/HLG video, and `--video-conditioning` accepts a conditioning video as either SDR or EXR frames.

### Changed

- Upgraded `ltx-core` to Transformers 5.8 or newer and added CUDA 13.2-compatible PyTorch, cuDNN, TorchCodec, NATTEN, and kernel-build dependency handling.
- Gemma loading now derives model structure from each local Hugging Face configuration, supports Gemma 3 and Gemma 4, and validates that the text encoder matches the LTX checkpoint.
- Gemma tokenization now consistently inserts a leading BOS token while avoiding duplicate BOS tokens for Gemma 3.
- Pipeline image-condition CRF defaults now come from checkpoint metadata: newer checkpoints use their trained value, while explicit CRF values remain unchanged.
- Pipeline constructors now accept a separate VAE checkpoint, diffusion-VAE optimization policy, prompt-enhancement model, and enhancement-cache setting.
- Direct construction of the former convolutional `VideoDecoder` now uses `ConvVideoDecoder`; `VideoDecoder` is now the common protocol implemented by convolutional and diffusion decoders.
- Trainer preprocessing, conditioning, validation, and latent decoding now derive spatial and temporal compression factors from checkpoint metadata instead of assuming 32x32x8.
- Trainer validation now uses separate video and audio CFG/STG controls, modality guidance, guidance rescaling, and checkpoint-aware frozen-modality handling.
- Trainer validation defaults now use 960x544x89 output, 24 fps, 30 inference steps, STG block 28, and a substantially expanded negative prompt.
- Legacy trainer validation guidance settings remain readable and are migrated automatically to their per-modality replacements.
- Model configurators now receive complete checkpoint metadata through `from_metadata`, enabling architecture and version-dependent construction.
- LoRA fusion and model reuse now preserve clean cached weights, avoid unnecessary tensor cloning, and support retained CPU weights.
- Transformer compilation modes that use CUDA graphs now require GPU-resident weights and fail early when used with incompatible offloading.

### Fixed

- Fixed multi-GPU video decoding applying pixel normalization twice, using a diffusion decoder's single-step forward path, and incorrectly rejecting temporal tiling configurations that do not actually split a worker's tile.
- Fixed multi-GPU prompt enhancement when encoding and enhancement use different Gemma roots, while retaining shared residency when they use the same root.
- Fixed Gemma 4 text encoding without a leading BOS token.
- Fixed trainer validation guidance for frozen audio or video, audio-only and video-only generation, cross-modal isolation, and modality-specific STG.
- Fixed trainer masks, spatial crops, reference-video alignment, and prefix/suffix validation for checkpoints with non-default VAE scale factors.
- Fixed LoRA fusion corrupting registry-cached weights and blockwise quantization failures when companion scale tensors remained on CPU.
- Fixed diffusion-VAE tiled decoding and multi-GPU blending to use checkpoint-specific geometry with lower peak host memory.
- Fixed 8-bit Gemma loading to resolve standard tokenizer assets and use architecture-agnostic Hugging Face model loading.
- Fixed CUDA builds using mismatched system toolkits or cuDNN sublibraries.
- Fixed diffusion-VAE decode tiling on Apple Silicon. The decode memory budget was probed only on CUDA, so on other backends it reported zero bytes available and no decode tile could ever fit, at any resolution. Automatic tiling now sizes itself from the Metal working set on MPS.

### Removed

- Removed `StateDictRegistry`; use `ModelRegistry`.
- Removed `SpatialTilingConfig` and `TemporalTilingConfig`; use `DimensionSizeConfig` with `TileSizeConfig`.
- Removed the old `TilingConfig` constructor fields `spatial_config` and `temporal_config`; use the new per-axis size/count tiling configuration.
- Removed `ltx_core.model.video_vae.tiling`; import tiling APIs from `ltx_core.tiling` or `ltx_core.model.video_vae`.
- Removed `GemmaTextEncoder`; use `LTXGemmaTextEncoder`.
- Removed `LTXVGemmaTokenizer`; use `LTXGemmaTokenizer`.
- Removed `GEMMA_LLM_KEY_OPS` and `GEMMA_MODEL_OPS`; use `get_gemma_ops()` for the selected Gemma root.
- Removed the hard-coded `Gemma3RopeScaling`, `Gemma3TextConfig`, `Gemma3VisionConfig`, `Gemma3ConfigData`, and `GEMMA3_CONFIG_FOR_LTX` definitions; Gemma configuration is now loaded from the model root through `gemma_model_config()`.
- Removed `ModelConfigurator.from_config()`; implement and call `from_metadata()`.
- Removed `DiffusionStage.model_context()` and `DiffusionStage.run()`; call the stage directly so it manages transformer construction and disposal.
- Removed `ltx_pipelines.utils.allocator_trim_strategy.AllocatorTrimStrategy`; import it from `ltx_core.allocator_trim_strategy`.
- Removed `ltx_trainer.training_strategies.VIDEO_SCALE_FACTORS`; use checkpoint-derived scale factors or `ltx_core.types.VIDEO_SCALE_FACTORS` when the legacy default is explicitly required.
- Removed trainer validation fields `guidance_scale`, `stg_scale`, and `stg_mode` from the current schema; use `video_cfg_scale`/`audio_cfg_scale` and `video_stg_scale`/`audio_stg_scale`. Legacy configuration files are migrated automatically.
- Removed the generic `gemma_i2v_system_prompt.txt` and `gemma_t2v_system_prompt.txt` names; use the `gemma3_*` or `gemma4_*` prompt files matching the encoder family.
- Removed `LipDubPipeline` and the `ltx_pipelines.lipdub` module; use `DubItPipeline` from `ltx_pipelines.dubit`.
