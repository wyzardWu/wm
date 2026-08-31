"""SF3 inference / action+strategy validation.

Loads a trained SF3 DiT checkpoint (full state dict, e.g. step-26000.safetensors)
and runs inference over the SF3 test split with prompt CFG enabled (cfg_scale=5.0
by default, since the user requested prompt-CFG validation).

Multi-GPU friendly: pass --gpu_ids "0,1,2,3" to round-robin clips across workers.
Outputs .mp4 + meta.json per clip plus a symlink to the original ground-truth clip
for side-by-side review.

Adapted from examples/sf2/inference/Wan2.2-TI2V-5B.py and examples/sf2_final/inference/infer_core.py.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/zeqingwang/zeqingwang/Training_Wan")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Make examples/sf3/data/ importable.
_REACTIVE_GWM = Path(__file__).resolve().parents[1]  # examples/ReactiveGWM/
if str(_REACTIVE_GWM) not in sys.path:
    sys.path.insert(0, str(_REACTIVE_GWM))

from data.profiles import SF3 as _PROFILE  # noqa: E402
SF3_BUTTON_COLS = list(_PROFILE.button_cols)
SF3_FIXED_PROMPT = _PROFILE.fixed_prompt
from data.action_utils import hold_last_upsample  # noqa: E402

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

WAN_MODEL_KWARGS = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48, "dim": 3072, "ffn_dim": 14336, "freq_dim": 256,
    "text_dim": 4096, "out_dim": 48, "num_heads": 24, "num_layers": 30,
    "eps": 1e-06,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}

BASE_MODEL_DIR = "/home/zeqingwang/zeqingwang/models/base_model"


@dataclass
class WorkerConfig:
    full_ckpt: str
    csv_path: str
    dataset_base: str
    output_dir: str
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    cfg_scale: float
    action_cfg_scale: float
    seed: int
    use_csv_prompt: bool
    action_hold_window: int


@dataclass
class Job:
    job_id: str
    video_rel: str
    action_rel: str
    prompt: str


def _transfer_weights(custom_model, pretrained_dit):
    pretrained = pretrained_dit.state_dict()
    custom = custom_model.state_dict()
    new = {k: v for k, v in pretrained.items()
           if k in custom and v.shape == custom[k].shape}
    custom_model.load_state_dict(new, strict=False)
    return custom_model


def _build_pipe(full_ckpt: str, device: str = "cuda"):
    import torch
    from safetensors.torch import load_file
    from diffsynth.pipelines.reactive_gwm import ReactiveGWMPipeline, ModelConfig
    from diffsynth.models.reactive_gwm_dit import ReactiveGWMModel

    pipe = ReactiveGWMPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[
            ModelConfig(local_model_path=BASE_MODEL_DIR, model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        download_source="huggingface"),
            ModelConfig(local_model_path=BASE_MODEL_DIR, model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        download_source="huggingface"),
            ModelConfig(local_model_path=BASE_MODEL_DIR, model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="Wan2.2_VAE.pth",
                        download_source="huggingface"),
        ],
        tokenizer_config=ModelConfig(
            local_model_path=BASE_MODEL_DIR, model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/", download_source="huggingface",
        ),
        redirect_common_files=False,
    )
    custom_dit = ReactiveGWMModel(num_buttons=len(SF3_BUTTON_COLS),
                                **WAN_MODEL_KWARGS).to(torch.bfloat16)
    pipe.dit = _transfer_weights(custom_dit, pipe.dit).to(pipe.device)
    del custom_dit
    torch.cuda.empty_cache()

    print(f"[ckpt] loading full DiT state dict: {full_ckpt}")
    state = load_file(full_ckpt)
    missing, unexpected = pipe.dit.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded {len(state)} keys (missing={len(missing)}, unexpected={len(unexpected)})")
    del state
    torch.cuda.empty_cache()
    return pipe


def _load_action(action_path: str, num_frames: int, hold_window: int, device: str = "cuda"):
    import torch
    df = pd.read_parquet(action_path)
    arr = df[SF3_BUTTON_COLS].values[:num_frames].astype(np.float32)
    keyboard = torch.tensor(arr)
    keyboard = hold_last_upsample(keyboard, window=hold_window)
    return keyboard.unsqueeze(0).to(device, torch.bfloat16)


def _worker(gpu_id: int, jobs: list, cfg: WorkerConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from diffsynth.utils.data import save_video, VideoData, crop_and_resize

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[gpu {gpu_id}] {len(jobs)} jobs; building pipe ...", flush=True)
    pipe = _build_pipe(cfg.full_ckpt, device="cuda")

    for i, job in enumerate(jobs):
        out_mp4 = out_root / f"{job.job_id}.mp4"
        if out_mp4.exists():
            print(f"[gpu {gpu_id}] [{i+1}/{len(jobs)}] skip (already done): {job.job_id}", flush=True)
            continue
        try:
            video_path = os.path.join(cfg.dataset_base, job.video_rel)
            action_path = os.path.join(cfg.dataset_base, job.action_rel)

            input_image = crop_and_resize(
                VideoData(video_path, height=cfg.height, width=cfg.width)[0],
                cfg.height, cfg.width,
            )
            keyboard = _load_action(action_path, cfg.num_frames,
                                    hold_window=cfg.action_hold_window, device="cuda")

            video = pipe(
                prompt=job.prompt,
                negative_prompt=NEGATIVE_PROMPT,
                input_image=input_image,
                num_frames=cfg.num_frames,
                num_inference_steps=cfg.num_inference_steps,
                cfg_scale=cfg.cfg_scale,
                action_cfg_scale=cfg.action_cfg_scale,
                height=cfg.height, width=cfg.width,
                seed=cfg.seed, tiled=True,
                keyboard_action=keyboard,
            )
            save_video(video, str(out_mp4), fps=20, quality=5)

            orig_link = out_root / f"{job.job_id}_original.mp4"
            if not orig_link.exists():
                try:
                    os.symlink(os.path.abspath(video_path), str(orig_link))
                except OSError:
                    pass

            meta = {
                "job_id": job.job_id,
                "video_rel": job.video_rel,
                "action_rel": job.action_rel,
                "prompt": job.prompt,
                "cfg_scale": cfg.cfg_scale,
                "action_cfg_scale": cfg.action_cfg_scale,
                "num_frames": cfg.num_frames,
                "num_inference_steps": cfg.num_inference_steps,
                "height": cfg.height, "width": cfg.width,
                "seed": cfg.seed,
                "ckpt": cfg.full_ckpt,
            }
            (out_root / f"{job.job_id}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            print(f"[gpu {gpu_id}] [{i+1}/{len(jobs)}] done: {job.job_id}", flush=True)
        except Exception as e:  # noqa: BLE001
            (out_root / f"{job.job_id}.error.txt").write_text(f"{e}\n\n{traceback.format_exc()}")
            print(f"[gpu {gpu_id}] [{i+1}/{len(jobs)}] FAIL {job.job_id}: {e}", flush=True)
        torch.cuda.empty_cache()


def _build_jobs(cfg: WorkerConfig, max_clips: Optional[int]) -> list:
    df = pd.read_csv(cfg.csv_path)
    if max_clips:
        df = df.iloc[:max_clips]
    jobs = []
    for _, row in df.iterrows():
        video_rel = row["video"]
        action_rel = row["action"]
        if cfg.use_csv_prompt and "prompt" in row and isinstance(row["prompt"], str) and row["prompt"].strip():
            prompt = row["prompt"]
        else:
            prompt = SF3_FIXED_PROMPT
        job_id = video_rel.replace("/video.mp4", "").replace("/", "_")
        jobs.append(Job(job_id=job_id, video_rel=video_rel, action_rel=action_rel, prompt=prompt))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_ckpt", type=str, required=True,
                        help="Path to full DiT state dict, e.g. step-26000.safetensors.")
    parser.add_argument("--csv_path", type=str,
                        default="/home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k/metadata_test.csv")
    parser.add_argument("--dataset_base", type=str,
                        default="/home/zeqingwang/zeqingwang/datasets/SF3/clips_10s_10k")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU ids, e.g. '0,1,2,3'")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=101)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="Prompt classifier-free guidance scale. >1.0 enables prompt CFG; "
                             "training used --prompt_dropout_prob 0.1 so cfg_scale~5.0 is the "
                             "intended evaluation knob.")
    parser.add_argument("--action_cfg_scale", type=float, default=1.0,
                        help="Action CFG (orthogonal to prompt CFG). 1.0 = disabled.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--use_csv_prompt", action="store_true",
                        help="Use per-row 'prompt' column (verbose NPC spec) instead of "
                             "SF3_FIXED_PROMPT. Match this to how the checkpoint was trained.")
    parser.add_argument("--action_hold_window", type=int, default=10,
                        help="Match training: 10 for hold-last upsample (default), 1 to disable.")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    if not gpu_ids:
        raise SystemExit("--gpu_ids must contain at least one GPU id")

    cfg = WorkerConfig(
        full_ckpt=args.full_ckpt,
        csv_path=args.csv_path,
        dataset_base=args.dataset_base,
        output_dir=args.output_dir,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        action_cfg_scale=args.action_cfg_scale,
        seed=args.seed,
        use_csv_prompt=args.use_csv_prompt,
        action_hold_window=args.action_hold_window,
    )

    jobs = _build_jobs(cfg, args.max_clips)
    print(f"=== SF3 inference: {len(jobs)} clips on GPUs {gpu_ids} ===")
    print(f"=== ckpt: {args.full_ckpt}")
    print(f"=== output: {args.output_dir}")
    print(f"=== prompt cfg_scale={args.cfg_scale}, action_cfg_scale={args.action_cfg_scale}")
    print(f"=== use_csv_prompt={args.use_csv_prompt}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "_run_args.json").write_text(json.dumps(vars(args), indent=2))

    if len(gpu_ids) == 1:
        _worker(gpu_ids[0], jobs, cfg)
        return

    buckets = [[] for _ in gpu_ids]
    for i, j in enumerate(jobs):
        buckets[i % len(gpu_ids)].append(j)
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, bucket in zip(gpu_ids, buckets):
        if not bucket:
            continue
        p = ctx.Process(target=_worker, args=(gpu_id, bucket, cfg))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
