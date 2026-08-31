"""ReactiveGWM video pipeline — Wan2.2-TI2V-5B + per-block action conditioning.

Slimmed-down fork of the Wan video pipeline that keeps only what the
ReactiveGWM training and inference paths need:

  • Single DiT (`pipe.dit`)              — Wan2.2-TI2V-5B is monolithic, no MoE.
  • T5 prompt encoder + tokenizer.
  • Wan2.2 VAE.
  • 5 pipeline units                     — ShapeChecker, NoiseInitializer,
                                           PromptEmbedder, InputVideoEmbedder,
                                           ImageEmbedderFused.
  • `model_fn_wan_video`                 — DiT forward with `keyboard_action`
                                           injection at each block + action CFG.

Removed (vs. the upstream wan_video.py the original SF2_final fork was based on):
  TeaCache, LongCat-Video, S2V (speech-to-video), Animate, VACE, VAP, Mot,
  Motion controller, FunControl/FunReference/FunCameraControl, CLIP image
  encoder, separate VAE-image embedder, CFG merger, USP (sequence parallel),
  sliding-window temporal tiler, dit2 / MoE switching, end-image, audio
  processor — none of these are exercised by ReactiveGWM training or inference.

The five remaining units are exactly the four cached-filtered units
(`WanVideoUnit_PromptEmbedder` / `_InputVideoEmbedder` / `_ImageEmbedderFused` /
`_NoiseInitializer`) plus `WanVideoUnit_ShapeChecker`. The training script
filters the four when `--use_cached_dataset` is set; ShapeChecker always runs.
"""
import torch
from PIL import Image
from einops import rearrange
from typing import Optional, Union
from typing_extensions import Literal
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE


class ReactiveGWMPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderFused(),
        ]
        self.model_fn = model_fn_wan_video


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        redirect_common_files: bool = True,
        vram_limit: float = None,
    ):
        # Redirect to converted-safetensors mirrors when the user passed the
        # legacy .pth filenames — saves a duplicate download.
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_t5_umt5-xxl-enc-bf16.safetensors",
                ),
                "Wan2.2_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.2_VAE.safetensors",
                ),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if (model_config.origin_file_pattern in redirect_dict
                        and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]):
                    print(
                        f"To avoid repeatedly downloading model files, "
                        f"({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to "
                        f"{redirect_dict[model_config.origin_file_pattern]}. "
                        f"You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]

        pipe = ReactiveGWMPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch the three models ReactiveGWM needs. The DiT loader accepts
        # index=2 (Wan2.x convention); for Wan2.2-TI2V-5B it returns a single
        # WanModel — if a list slips through (mis-configured Wan2.2 MoE), we
        # train against the first expert and warn the user.
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            print(f"[ReactiveGWMPipeline] DiT loader returned {len(dit)} experts; "
                  "ReactiveGWM trains a single DiT, taking expert 0.")
            pipe.dit = dit[0]
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")

        # Size division factor: VAE downsamples 8x but DiT also patches by 2x,
        # so the H/W must be divisible by 16 (Wan2.2 VAE upsampling_factor=16).
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path, seq_len=512, clean='whitespace',
            )

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video / video-to-video
        input_image: Optional[Image.Image] = None,
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        # Classifier-free guidance (text)
        cfg_scale: Optional[float] = 5.0,
        # Action-conditioning CFG (orthogonal to text CFG).
        # When != 1.0, do an extra forward with keyboard_action zeroed out and
        # amplify the action contribution: eps = eps_noact + w * (eps_cond - eps_noact).
        action_cfg_scale: Optional[float] = 1.0,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # progress
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
        # Action — the ReactiveGWM-specific input.
        keyboard_action: Optional[torch.Tensor] = None,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps,
                                     denoising_strength=denoising_strength,
                                     shift=sigma_shift)

        # Inputs are split three ways for the unit pipeline:
        #   posi  — keyed off `prompt`     (positive CFG branch)
        #   nega  — keyed off `negative_prompt`  (negative CFG branch)
        #   shared — single-branch tensors
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_image": input_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "keyboard_action": keyboard_action,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega,
            )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Conditional forward (prompt + action).
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)

            # Action CFG: rerun the posi branch with keyboard_action zeroed
            # to get the "unconditional w.r.t. action" prediction, then
            # amplify the delta. Requires a real keyboard_action.
            if action_cfg_scale != 1.0 and inputs_shared.get("keyboard_action") is not None:
                _ka_backup = inputs_shared["keyboard_action"]
                inputs_shared["keyboard_action"] = torch.zeros_like(_ka_backup)
                noise_pred_posi_noact = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
                inputs_shared["keyboard_action"] = _ka_backup
                noise_pred_posi = noise_pred_posi_noact + action_cfg_scale * (noise_pred_posi - noise_pred_posi_noact)

            # Text CFG.
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler step + first-frame latent pin (keeps the input image's
            # encoded latent locked across the trajectory).
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(
            inputs_shared["latents"], device=self.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: ReactiveGWMPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: ReactiveGWMPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1, pipe.vae.model.z_dim, length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}



class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: ReactiveGWMPipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            input_video, device=pipe.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            # Training feeds noise + input_latents to FlowMatchSFTLoss separately;
            # the loss adds noise via the scheduler, so latents starts as noise.
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe: ReactiveGWMPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: ReactiveGWMPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode the input first frame to a latent and overwrite latents[:, :, 0:1].

    Specific to Wan2.2-TI2V-5B (`fuse_vae_embedding_in_latents=True`); the I2V
    conditioning rides on the first latent slot rather than as an extra `y`
    channel concat.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width",
                          "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: ReactiveGWMPipeline, input_image, latents,
                height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode(
            [image], device=pipe.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    keyboard_action: Optional[torch.Tensor] = None,
    **kwargs,
):
    """ReactiveGWM DiT forward.

    Two ReactiveGWM-specific deviations from a vanilla Wan DiT forward:

      1. `fuse_vae_embedding_in_latents=True` triggers the Wan2.2-TI2V-5B
         "first frame is timestep-0, all others share `timestep`" pattern:
         we build a per-token timestep tensor with zeros on frame 0 spatial
         positions before time embedding.
      2. `keyboard_action` is downsampled ONCE here via
         `dit.prepare_action_binned(...)` and then injected at every block by
         `dit.inject_at_block(...)` (see ReactiveGWMModel docstring for why
         per-block embedders + no LayerScale gate).

    The function tolerates extra kwargs (`**kwargs`) so dict-splat callers in
    the pipeline can pass shared inputs blindly without having to whitelist.
    """
    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4),
                        dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                       dtype=latents.dtype, device=latents.device) * timestep,
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents
    # Patchify (no camera-control input — ReactiveGWMModel ignores that branch)
    x = dit.patchify(x)

    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    # Action embedding prep (ReactiveGWMModel v4: per-block independent
    # embedders). Downsample keyboard_action to latent frame rate ONCE; each
    # DiTBlock below runs the shared binned tensor through its own nn.Linear.
    action_binned = None
    action_spatial = None  # (f, h, w) at action-embed time
    if keyboard_action is not None and hasattr(dit, 'prepare_action_binned'):
        action_binned = dit.prepare_action_binned(keyboard_action, f)
        action_spatial = (f, h, w)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # Blocks
    for block_id, block in enumerate(dit.blocks):
        # ReactiveGWMModel v4: per-block independent action embedder. Each
        # DiTBlock projects action_binned through its own nn.Linear and adds
        # the result to x. No LayerScale gate (see ReactiveGWMModel docstring
        # for the bf16 precision rationale).
        if action_binned is not None and hasattr(dit, 'inject_at_block'):
            f_a, h_a, w_a = action_spatial
            x = dit.inject_at_block(x, action_binned, block_id, f_a, h_a, w_a)
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x, context, t_mod, freqs,
        )

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
