"""Multiprocessing runner shared by eval_action / eval_strategy / per-game extras.

Each worker process:
  1. Pins itself to one GPU via CUDA_VISIBLE_DEVICES (set BEFORE torch import).
  2. Builds a single ReactiveGWMPipeline.
  3. Runs assigned jobs sequentially.

Jobs are distributed round-robin across workers. The active GameProfile is
passed by *name* (string) — workers re-resolve it after CUDA pinning so the
profile object never has to cross the spawn boundary.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    """One inference job."""
    job_id: str                        # unique, used for output filenames
    prompt: str
    action_spec: dict                  # opaque dict consumed by `build_action_tensor`
    input_image_path: str              # absolute path to the source clip's video.mp4
    seed: int = 0
    extra: dict = field(default_factory=dict)  # per-job metadata to dump in meta.json


@dataclass
class WorkerConfig:
    """Static config shared by every worker."""
    base_model_dir: str
    full_ckpt: str | None
    lora_ckpt: str | None
    lora_alpha: float
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    cfg_scale: float
    action_cfg_scale: float
    output_dir: str
    game: str = "sf2"   # profile name; resolved inside each worker


def _worker(gpu_id: int, jobs: list[Job], cfg: WorkerConfig, reactive_gwm_dir: str):
    """Worker entry point — runs in spawned subprocess."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if reactive_gwm_dir not in sys.path:
        sys.path.insert(0, reactive_gwm_dir)
    # Import after CUDA pinning so torch sees only the assigned GPU.
    import torch  # noqa: F401  (also forces CUDA init under correct visibility)
    from data.profiles import get_profile
    from diffsynth.utils.data import save_video
    from inference.infer_core import (
        CkptSpec, build_pipe, first_frame_pil, run_inference,
    )
    from inference.metrics import frames_from_pil_list, region_motion_energy

    profile = get_profile(cfg.game)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[worker gpu={gpu_id} game={profile.name}] {len(jobs)} jobs; building pipe...")
    pipe = build_pipe(
        profile,
        cfg.base_model_dir,
        CkptSpec(full_ckpt=cfg.full_ckpt, lora_ckpt=cfg.lora_ckpt,
                 lora_alpha=cfg.lora_alpha),
        device="cuda",
    )

    for i, job in enumerate(jobs):
        try:
            input_image = first_frame_pil(job.input_image_path, cfg.height, cfg.width)
            keyboard = build_action_tensor(profile, job.action_spec, cfg.num_frames, device="cuda")
            video = run_inference(
                pipe=pipe,
                prompt=job.prompt,
                keyboard_action=keyboard,
                input_image=input_image,
                num_frames=cfg.num_frames,
                height=cfg.height, width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                cfg_scale=cfg.cfg_scale,
                action_cfg_scale=cfg.action_cfg_scale,
                seed=job.seed,
            )
            mp4 = out_root / f"{job.job_id}.mp4"
            save_video(video, str(mp4), fps=20, quality=5)
            frames = frames_from_pil_list(video)
            metrics = {
                "me_full":  region_motion_energy(frames, "full"),
                "me_left":  region_motion_energy(frames, "left"),
                "me_right": region_motion_energy(frames, "right"),
            }
            meta = {
                "job_id": job.job_id,
                "prompt": job.prompt,
                "action_spec": job.action_spec,
                "input_image_path": job.input_image_path,
                "seed": job.seed,
                "metrics": metrics,
                "extra": job.extra,
            }
            (out_root / f"{job.job_id}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False)
            )
            print(f"[worker gpu={gpu_id}] {i+1}/{len(jobs)} done: {job.job_id}")
        except Exception as e:  # noqa: BLE001 — never let one job kill the worker
            err = (out_root / f"{job.job_id}.error.txt")
            err.write_text(f"{e}\n\n{traceback.format_exc()}")
            print(f"[worker gpu={gpu_id}] FAIL {job.job_id}: {e}")
        import torch as _t
        _t.cuda.empty_cache()


def build_action_tensor(profile, spec: dict, num_frames: int, device: str = "cuda"):
    """Decode an action_spec dict into a [1, T, num_buttons] keyboard tensor.

    Two spec formats:
      {"kind": "constant", "buttons": ["LEFT"]}              -> all frames pressed
      {"kind": "segments", "segments": [["LEFT"], ["A"], ["UP","RIGHT"], []]}
                                                                -> equal-length segments
    Button names must be in `profile.button_cols`. SF2 and SF3 share the same
    10 button names; the *button-strength* mapping is a caller concern.
    """
    import torch
    import numpy as np
    name_to_idx = {n: i for i, n in enumerate(profile.button_cols)}
    arr = np.zeros((num_frames, profile.num_buttons), dtype=np.float32)
    if spec["kind"] == "constant":
        for b in spec["buttons"]:
            arr[:, name_to_idx[b]] = 1.0
    elif spec["kind"] == "segments":
        segs = spec["segments"]
        n = len(segs)
        seg_len = num_frames // n
        for k, btns in enumerate(segs):
            start = k * seg_len
            end = (k + 1) * seg_len if k < n - 1 else num_frames
            for b in btns:
                arr[start:end, name_to_idx[b]] = 1.0
    else:
        raise ValueError(spec["kind"])
    return torch.tensor(arr).unsqueeze(0).to(device, torch.bfloat16)


def run_jobs(jobs: list[Job], cfg: WorkerConfig, gpu_ids: list[int],
             reactive_gwm_dir: str) -> None:
    """Distribute jobs round-robin and run them in parallel on `gpu_ids`."""
    if not gpu_ids:
        raise ValueError("Need at least one GPU id")
    buckets: list[list[Job]] = [[] for _ in gpu_ids]
    for i, j in enumerate(jobs):
        buckets[i % len(gpu_ids)].append(j)
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, bucket in zip(gpu_ids, buckets):
        if not bucket:
            continue
        p = ctx.Process(target=_worker, args=(gpu_id, bucket, cfg, reactive_gwm_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
