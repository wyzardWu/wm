"""On-protocol AR KV-cache rollout for stage1 causal AR-diffusion checkpoints.

Faithfully mirrors zhiyang's deploy path (dmd_gm.py self_forcing_rollout, absolute
KV mode): commit clean frame0 -> per frame k: full Euler denoise with commit=False
transient passes -> exactly one commit=True pass with t=0. Uses his GameMasterDiT /
FlowMatchScheduler verbatim; only this driver is ours.

Usage:
  python scripts/s1_ar_rollout.py --dit dits/dit_step500.safetensors \
      --image bridge_seed.png --actions W2s_stop3s.parquet \
      --table action_context_table_v3.pt --out out.mp4 [--steps 30] [--seed 2]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ReactiveGWM"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cf_distill_3stage"))

from PIL import Image
from safetensors.torch import load_file

from ReactiveGWM_Code.inference.models import WanVideoVAE38
from ReactiveGWM_Code.inference.utils import preprocess_image, to_pil_video, save_mp4
from gamemaster.models.gamemaster_dit import GameMasterDiT, ti2v_5b_config
from gamemaster.flow_match import FlowMatchScheduler

COLS5 = ["W", "A", "S", "D", "MOUSE0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--actions", required=True, help="parquet, 4k+1 rows")
    ap.add_argument("--table", required=True)
    ap.add_argument("--scene_table", default=None,
                    help="EYBX: [n_scenes+1,16,4096] scene table; parquet must "
                         "carry a SCENE column (ids, null = n_scenes). Context "
                         "becomes cat(action_row, scene_row) [32,4096].")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--kv_window", type=int, default=0,
                    help=">0 enables zhiyang's SLIDING KV mode for stable long rollout")
    ap.add_argument("--rope_cap", type=int, default=16)
    ap.add_argument("--sink_size", type=int, default=1)
    ap.add_argument("--base", default="/nfs/zeqingwang/models/base_model")
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16

    # --- VAE (same load path as SFPipeline.from_pretrained) ---
    vae = WanVideoVAE38()
    vs = torch.load(Path(args.base) / "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
                    map_location="cpu", weights_only=True)
    miss, unexp = vae.load_state_dict({f"model.{k}": v for k, v in vs.items()}, strict=False)
    assert not unexp, unexp[:5]
    vae = vae.to(dt).eval().requires_grad_(False).to(dev)

    # --- causal DiT (stage1 arch: ti2v_5b_config, causal=True) ---
    model = GameMasterDiT(**ti2v_5b_config(), causal=True, zero_init_head=False)
    sd = load_file(args.dit)
    info = model.load_state_dict(sd, strict=False)
    assert not info.missing_keys and not info.unexpected_keys, \
        (info.missing_keys[:5], info.unexpected_keys[:5])
    model = model.to(dt).eval().requires_grad_(False).to(dev)
    print(f"[ar] dit loaded: {len(sd)} keys from {args.dit}", flush=True)

    # --- per-latent-frame contexts (identical binning to SFPipeline table mode) ---
    tbl = torch.load(args.table, map_location="cpu", weights_only=False)
    table = tbl["table"] if isinstance(tbl, dict) else tbl          # [33,16,4096]
    df = pd.read_parquet(args.actions)
    kb = torch.tensor(df[COLS5].values, dtype=torch.float32)[None]  # [1,N,5]
    f_lat = (len(df) - 1) // 4 + 1
    binned = F.adaptive_max_pool1d(kb.transpose(1, 2), f_lat).transpose(1, 2)
    b = (binned > 0.5).long()
    ids = (b[..., 0] * 1 + b[..., 1] * 2 + b[..., 2] * 4 + b[..., 3] * 8 + b[..., 4] * 16)[0]
    ctx_all = table[ids].to(dev, dt)[None]                          # [1,f_lat,16,4096]
    if args.scene_table:
        stbl_blob = torch.load(args.scene_table, map_location="cpu", weights_only=False)
        stbl = stbl_blob["table"] if isinstance(stbl_blob, dict) else stbl_blob
        skb = torch.tensor(df["SCENE"].values, dtype=torch.float32)[None, :, None]
        sbin = F.adaptive_max_pool1d(skb.transpose(1, 2), f_lat).transpose(1, 2)
        sids = sbin[0, :, 0].round().long().clamp(0, stbl.shape[0] - 1)
        ctx_all = torch.cat([ctx_all, stbl[sids].to(dev, dt)[None]], dim=2)  # [1,f_lat,32,4096]
        print(f"[ar] scene ids: {sids.tolist()}", flush=True)
    print(f"[ar] {len(df)} action rows -> {f_lat} latent frames, ids={ids.tolist()[:10]}...",
          flush=True)

    # --- seed latent (TI2V fuse convention) ---
    img_t = preprocess_image(Image.open(args.image), 480, 832, device=dev, dtype=dt)
    lat0 = vae.encode([img_t], device=dev).to(dev, dt)              # [1,48,1,h,w]

    # --- AR rollout: mirrors dmd_gm self_forcing_rollout, full-step diffusion ---
    sched = FlowMatchScheduler(shift=5.0)
    sched.set_timesteps(args.steps, device=dev)
    if args.kv_window > 0:
        cache = model.init_kv_cache(kv_window=args.kv_window,
                                    rope_cap=args.rope_cap, sink_size=args.sink_size)
        print(f"[ar] SLIDING KV: window={args.kv_window} rope_cap={args.rope_cap} "
              f"sink={args.sink_size}", flush=True)
    else:
        cache = model.init_kv_cache()
    zero_t = torch.zeros(1, device=dev)
    gen = torch.Generator("cpu").manual_seed(args.seed)
    frames = [lat0]
    with torch.no_grad():
        model(lat0, zero_t, ctx_all[:, 0:1], frame_index=0, kv_cache=cache, commit=True)
        for k in range(1, f_lat):
            x = torch.randn((1, 48, 1, lat0.shape[3], lat0.shape[4]),
                            generator=gen, dtype=torch.float32).to(dev, dt)
            for i in range(args.steps):
                t = sched.infer_timesteps[i].reshape(1).to(dev)
                v = model(x, t, ctx_all[:, k:k + 1], frame_index=k,
                          kv_cache=cache, commit=False)
                x = sched.step(v, x, i)
            model(x, zero_t, ctx_all[:, k:k + 1], frame_index=k,
                  kv_cache=cache, commit=True)
            frames.append(x)
            if k % 5 == 0 or k == f_lat - 1:
                print(f"[ar] frame {k}/{f_lat - 1} committed", flush=True)
        lat = torch.cat(frames, dim=2)                              # [1,48,f_lat,h,w]
        video = vae.decode(lat, device=dev)
    save_mp4(to_pil_video(video), args.out, fps=20)
    print(f"[ar] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
