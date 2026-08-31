"""Shim for zhiyang's vendor `modules.vae2_2.Wan2_2_VAE` (dmd_gm re-anchor).
Same constructor/API, backed by ReactiveGWM's WanVideoVAE38 (identical Wan2.2 VAE
weights). encode/decode are list-in/list-out, matching the vendor wrapper."""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "ReactiveGWM"))
sys.path.insert(0, str(_ROOT))

from ReactiveGWM_Code.inference.models import WanVideoVAE38


class Wan2_2_VAE:
    def __init__(self, vae_pth, device="cuda", dtype=torch.bfloat16):
        vae = WanVideoVAE38()
        state = torch.load(vae_pth, map_location="cpu", weights_only=True)
        miss, unexp = vae.load_state_dict({f"model.{k}": v for k, v in state.items()},
                                          strict=False)
        assert not unexp, unexp[:5]
        self.vae = vae.to(dtype).eval().requires_grad_(False).to(device)
        self.device = device

    def encode(self, videos):
        return self.vae.encode([v.to(self.device) for v in videos], device=self.device)

    def decode(self, latents):
        return self.vae.decode([l.to(self.device) for l in latents], device=self.device)
