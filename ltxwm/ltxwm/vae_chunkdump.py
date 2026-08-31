"""VAE encode->decode roundtrip check for the 2.3 monolith in this env.
If the roundtrip is clean, VAE loading/usage is fine and the noise bug is
upstream (denoiser); if the roundtrip is noise, the VAE path is broken."""
import sys
import torch

sys.path.insert(0, "/data/yuzhewu/ltxwm")
CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
CLIP = "/nfs/danze/data/abot/clips/6b/6b489fcc806d7d56bae96c595a0d4bd6_w000.mp4"
OUT = "/data/yuzhewu/vrisingroam/ltx_probes/vae_roundtrip.mp4"

from ltx_pipelines.utils.blocks import ImageConditioner, VideoDecoder
from ltx_pipelines.utils.helpers import ensure_tiling_config, tiling_scale_factors_for_vae
from ltx_core.types import VideoPixelShape
from ltx_core.model.video_vae import AUTO_TILING

import imageio.v3 as iio
import numpy as np

dev = torch.device("cuda")
frames = iio.imread(CLIP, index=None)[:65]           # (F,H,W,C) uint8 480x832
vid = torch.from_numpy(frames).float().div(127.5).sub(1.0)
vid = vid.permute(3, 0, 1, 2).unsqueeze(0).to(dev, torch.bfloat16)  # [1,C,F,H,W]
print("pixels", vid.shape, flush=True)

ic = ImageConditioner(CKPT, dtype=torch.bfloat16, device=dev)
holder = {}
with torch.inference_mode():
    ic(lambda enc: holder.update(lat=enc(vid)) or [])
lat = holder["lat"]
print("latent", lat.shape, float(lat.float().mean()), float(lat.float().std()), flush=True)

vd = VideoDecoder(CKPT, dtype=torch.bfloat16, device=dev)
scale = tiling_scale_factors_for_vae(vd.checkpoint_path)
tc = ensure_tiling_config(
    AUTO_TILING, scale_factors=scale, vae_checkpoint_path=vd.checkpoint_path,
    video_shape=VideoPixelShape(batch=1, frames=65, height=480, width=832, fps=24.0),
    diffvae_optimization=vd.diffvae_optimization, device=dev,
)
with torch.inference_mode():
    gen = torch.Generator(device=dev).manual_seed(0)
    chunks = vd(lat, tc, generator=gen, dtype=torch.bfloat16)
    first = next(iter(chunks))
    print("chunk", tuple(first.shape), first.dtype,
          float(first.float().min()), float(first.float().max()),
          float(first.float().mean()), flush=True)
    fr = first
    while fr.ndim > 3:
        fr = fr[0]
    # try to interpret as HWC
    a = fr.float().clamp(0,1).mul(255).byte().cpu().numpy()
    print("frame0 slice shape", a.shape, flush=True)
    from PIL import Image
    if a.ndim == 3 and a.shape[-1] in (3,4):
        Image.fromarray(a[..., :3]).save(OUT + ".frame0.png")
    elif a.ndim == 3 and a.shape[0] in (3,4):
        Image.fromarray(a[:3].transpose(1,2,0)).save(OUT + ".frame0.png")
    print("saved frame0 png", flush=True)
