"""Action-injection behavioral probe for LTX-2.3 + smoke LoRA.

Same first frame + same seed, different action scripts -> compare motion.
Reuses the official TI2VidOneStagePipeline with three surgical swaps:
  1. loras=[trainer LoRA] via the native loader
  2. stage._model_wrapper installs the frame-fold patch on the built model
  3. prompt_encoder stubbed: contexts come from the clip's precomputed caption
     embeds + our action tables (per-frame [caption 1024 | move 32 | view 32]);
     negative context = caption + idle actions (same length -> CFG works)

Usage:
  python probe_action.py --clip_idx 0 --probe W_hold --out /path/out.mp4
  probes: still | W_hold | S_hold | A_hold | D_hold | W_J | switch_WS
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/data/yuzhewu/ltxwm")

CKPT = "/data/yuzhewu/ltxwm/ltx-2.3-22b-dev.safetensors"
GEMMA = "/data/yuzhewu/ltxwm/gemma-3-12b-it"
TABLES = "/data/yuzhewu/ltxwm/tables/ltx_action_tables.pt"
PRECOMP = "/data/yuzhewu/ltxwm/data/train2000_precomp"
MANIFEST = "/data/yuzhewu/ltxwm/data/train2000.jsonl"
CAP_LEN, ACT_LEN = 1024, 64
MOVE_ID = {k: i for i, k in enumerate(["", "W", "S", "A", "D", "WA", "WD", "SA", "SD"])}
VIEW_ID = {k: i for i, k in enumerate(["", "J", "L", "I", "K", "JI", "JK", "LI", "LK"])}

PROBES = {
    "still":     lambda f: ("", ""),
    "W_hold":    lambda f: ("W", ""),
    "S_hold":    lambda f: ("S", ""),
    "A_hold":    lambda f: ("A", ""),
    "D_hold":    lambda f: ("D", ""),
    "W_J":       lambda f: ("W", "J"),
    "A_L":       lambda f: ("A", "L"),
    "I_hold":    lambda f: ("", "I"),
    "D_I":       lambda f: ("D", "I"),
    "switch_WS": lambda f: ("W" if f < 5 else "S", ""),
    # 四方向 + 中段少量相机(潜帧 6-9 ≈1.3s 左转),四宫格科目
    "W_camJ": lambda f: ("W", "J" if 6 <= f < 10 else ""),
    "S_camJ": lambda f: ("S", "J" if 6 <= f < 10 else ""),
    "A_camJ": lambda f: ("A", "J" if 6 <= f < 10 else ""),
    "D_camJ": lambda f: ("D", "J" if 6 <= f < 10 else ""),
}


class StubOutput:
    def __init__(self, video_encoding, audio_encoding, attention_mask):
        self.video_encoding = video_encoding
        self.audio_encoding = audio_encoding
        self.attention_mask = attention_mask


def build_ctx(cap, mt, vt, fn, F):
    rows = []
    for f in range(F):
        mv, vw = fn(f)
        rows.append(torch.cat([cap, mt[MOVE_ID[mv]], vt[VIEW_ID[vw]]], dim=0))
    return torch.cat(rows, dim=0).unsqueeze(0)          # [1, F*(1024+64), D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_idx", type=int, default=0)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--precomp", default=PRECOMP)
    ap.add_argument("--tables", default=TABLES)
    ap.add_argument("--probe", required=True, choices=sorted(PROBES))
    ap.add_argument("--lora", default="/data/yuzhewu/ltxwm/runs/abot2000_v1/checkpoints/lora_weights_step_00500.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=65, help="pixel frames, 8k+1")
    ap.add_argument("--baseline", action="store_true",
                    help="pure official path: no lora, no fold, real prompt encoder")
    ap.add_argument("--inject", choices=["fold", "adaln"], default="fold",
                    help="adaln: 离散 id AdaLN 臂(caption 全局 CA,动作走 adaln_single)")
    ap.add_argument("--adaln_ckpt", default=None,
                    help="action_adaln_step_XXXXX.pt (embedder state_dict), adaln 模式必填")
    ap.add_argument("--no_lora", action="store_true",
                    help="keep fold+stub+action context but load NO lora (base-capability control)")
    ap.add_argument("--neg_probe", default="still",
                    help="probe name used for the CFG negative context (default still)")
    args = ap.parse_args()

    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltxwm.frame_context_patch import install_frame_context

    row = [json.loads(line) for line in open(args.manifest)][args.clip_idx]
    rel = row["video"].lstrip("/").replace(".mp4", ".pt")
    cond = torch.load(os.path.join(args.precomp, "conditions", rel), map_location="cpu", weights_only=True)
    cap_raw = cond["video_prompt_embeds"].to(torch.bfloat16)      # [1024, 4096] PRE-connector
    aud_raw = cond["audio_prompt_embeds"].to(torch.bfloat16)      # [1024, 2048] PRE-connector

    # Precomputed embeds are PRE-connector (the trainer runs create_embeddings on
    # them every step). Mirror that here so the probe context matches training.
    sys.path.insert(0, "/data/yuzhewu/ltxwm/LTX-2/packages/ltx-trainer/src")
    from ltx_trainer.model_loader import load_embeddings_processor, embedding_weight_paths
    from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask
    proc = load_embeddings_processor(
        checkpoint_path=embedding_weight_paths(CKPT, GEMMA),
        gemma_model_path=GEMMA, device="cpu", dtype=torch.bfloat16)
    pmask = cond["prompt_attention_mask"].unsqueeze(0)
    additive = convert_to_additive_mask(pmask, torch.bfloat16)
    with torch.inference_mode():
        cap_b, aud_b, _ = proc.create_embeddings(
            cap_raw.unsqueeze(0), aud_raw.unsqueeze(0), additive)
    cap, aud = cap_b[0], aud_b[0]
    del proc

    blob = torch.load(args.tables, map_location="cpu", weights_only=True)
    mt, vt = blob["move_table"].to(cap.dtype), blob["view_table"].to(cap.dtype)

    # first frame png from the source clip
    import imageio.v3 as iio
    from PIL import Image
    frame0 = iio.imread(row["video"], index=0)
    seed_png = args.out + ".seed.png"
    Image.fromarray(frame0).save(seed_png)

    F = (args.frames - 1) // 8 + 1
    mp = ModelPaths.from_monolith(checkpoint_path=CKPT, gemma_root=GEMMA)
    from ltx_pipelines.utils.types import OffloadMode
    pipe = TI2VidOneStagePipeline(
        model_paths=mp,
        loras=[] if (args.baseline or args.no_lora) else [LoraPathStrengthAndSDOps(args.lora, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)],
        offload_mode=OffloadMode.CPU,
    )

    if not args.baseline and args.inject == "adaln":
        # AdaLN 臂:caption 走原生全局 CA(pos=neg=caption),动作差异全在 ids。
        # denoiser 把 [cond, uncond] 拼成一个 batch => ids 顺序 [探针, neg_probe]。
        from ltxwm.action_adaln_patch import HOLDER, install_action_adaln
        assert args.adaln_ckpt, "--adaln_ckpt required for --inject adaln"

        def ids_of(fn):
            return torch.tensor([[MOVE_ID[fn(f)[0]], VIEW_ID[fn(f)[1]]] for f in range(F)],
                                dtype=torch.long)
        HOLDER["ids"] = torch.stack([ids_of(PROBES[args.probe]),
                                     ids_of(PROBES[args.neg_probe])])   # [2, F, 2]
        HOLDER["seq_cfg"] = True   # one-stage 管线 CFG 为串行两次 forward

        def wrapper(model, _tools=None):
            emb = install_action_adaln(model)
            sd = torch.load(args.adaln_ckpt, map_location="cpu", weights_only=True)
            emb.load_state_dict(sd)
            emb.to(device="cuda", dtype=torch.float32)
            HOLDER["module"] = emb
            print(f"[probe] adaln embedder loaded: {args.adaln_ckpt}", flush=True)
            return model
        pipe.stage._model_wrapper = wrapper

        dev = pipe.device
        cap1 = cap.unsqueeze(0).to(dev)
        a_ctx = aud.unsqueeze(0).to(dev)
        ones = lambda L: torch.ones(1, L, dtype=torch.int64, device=dev)

        def stub(prompts, **kw):
            return [
                StubOutput(cap1, a_ctx, ones(cap1.shape[1])),
                StubOutput(cap1, a_ctx, ones(cap1.shape[1])),
            ]
        pipe.prompt_encoder = stub
    elif not args.baseline:
        # frame-fold via the official model wrapper hook
        def wrapper(model, _tools=None):
            n = install_frame_context(model, F, CAP_LEN + ACT_LEN)
            print(f"[probe] frame-fold on {n} blocks", flush=True)
            return model
        pipe.stage._model_wrapper = wrapper

        dev = pipe.device
        pos = build_ctx(cap, mt, vt, PROBES[args.probe], F).to(dev)
        neg = build_ctx(cap, mt, vt, PROBES[args.neg_probe], F).to(dev)
        a_ctx = aud.unsqueeze(0).to(dev)
        ones = lambda L: torch.ones(1, L, dtype=torch.int64, device=dev)

        def stub(prompts, **kw):
            return [
                StubOutput(pos, a_ctx, ones(pos.shape[1])),
                StubOutput(neg, a_ctx, ones(neg.shape[1])),
            ]
        pipe.prompt_encoder = stub

    prompt = row["caption"] if args.baseline else "stub"
    with torch.inference_mode():
        out = pipe(
        prompt=prompt, negative_prompt="worst quality, static, blurry",
        seed=args.seed, height=480, width=832, frame_rate=24.0,
        num_inference_steps=args.steps,
        video_guider_params=MultiModalGuiderParams(cfg_scale=args.cfg),
        audio_guider_params=MultiModalGuiderParams(cfg_scale=1.0),
        images=[ImageConditioningInput(seed_png, 0, 1.0)],
        num_frames=args.frames,
    )
    from ltx_pipelines.utils.media_io import encode_video
    with torch.inference_mode():
        encode_video(out.video, fps=24, audio=None, output_path=args.out,
                     video_chunks_number=out.num_frames)
    print("[probe] saved", args.out, flush=True)


if __name__ == "__main__":
    main()
