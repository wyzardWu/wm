"""Post-swap VAE sanity gate: encode->decode a real clip, require natural-image
statistics (adjacent-pixel correlation) in the output. Corrupt weights give
uncorrelated speckle (~0); real decodes give >0.9. Exit 0 = pass, 1 = fail."""
import sys
import torch

sys.path.insert(0, "/data/yuzhewu/ltxwm")
CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
CLIP = "/nfs/danze/data/abot/clips/6b/6b489fcc806d7d56bae96c595a0d4bd6_w000.mp4"
OUT = "/data/yuzhewu/vrisingroam/ltx_probes/vae_gate_frame0.png"

from ltx_pipelines.utils.blocks import ImageConditioner, VideoDecoder
import imageio.v3 as iio

dev = torch.device("cuda")
fr = iio.imread(CLIP, index=None)[:65]
vid = torch.from_numpy(fr).float().div(127.5).sub(1).permute(3, 0, 1, 2).unsqueeze(0).to(dev, torch.bfloat16)
ic = ImageConditioner(CKPT, dtype=torch.bfloat16, device=dev)
h = {}
with torch.inference_mode():
    ic(lambda enc: h.update(lat=enc(vid)) or [])
lat = h["lat"]
print("latent", tuple(lat.shape), float(lat.float().mean()), float(lat.float().std()), flush=True)
vd = VideoDecoder(CKPT, dtype=torch.bfloat16, device=dev)
with torch.inference_mode():
    it = iter(vd(lat, None, generator=torch.Generator(device=dev).manual_seed(0)))
    first = next(it)
    for _ in it:
        pass
img = first[0].float()  # (H, W, 3) in [0,1]
corr = torch.corrcoef(torch.stack([img[:, :-1].flatten(), img[:, 1:].flatten()]))[0, 1].item()
src = torch.from_numpy(fr[0]).float().div(255)
psnr = -10 * torch.log10(((img.cpu() - src) ** 2).mean()).item()
print(f"adjacent-pixel corr={corr:.4f} psnr={psnr:.2f}dB", flush=True)
from PIL import Image
Image.fromarray(img.clamp(0, 1).mul(255).byte().cpu().numpy()).save(OUT)
print("saved", OUT, flush=True)
if corr < 0.8 or psnr < 20:
    print("GATE FAIL", flush=True)
    sys.exit(1)
print("GATE PASS", flush=True)
