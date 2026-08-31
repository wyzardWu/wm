"""Non-blocking subprocess spawn for sanity-check inference.

被 train.py 在每个 save_steps 调用; 训练 main loop **永不阻塞**.
若上一次 subprocess 仍 alive 而本次 spawn 又触发, log warning 并 skip 本次.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any


def spawn_sanity_sample(
    *,
    ckpt_path: str,
    step: int,
    out_dir: str,
    cfg: dict[str, Any],
    config_path: str,
    last_proc: subprocess.Popen | None = None,
) -> subprocess.Popen | None:
    """Spawn sanity_sample.py as a non-blocking subprocess and return its Popen.

    Args:
        ckpt_path: latest model safetensors (the one we just dumped).
        step: training step (used for output filename).
        out_dir: training output dir; mp4 + log go under `<out_dir>/samples/`.
        cfg: parsed config dict (used to pick device etc).
        config_path: yaml path so the subprocess can re-read full config.
        last_proc: previous Popen handle; if still alive we skip.

    Returns the new Popen handle (or `last_proc` if skipped).
    """
    if last_proc is not None and last_proc.poll() is None:
        print(
            f"[Sanity-Spawn] previous subprocess (pid={last_proc.pid}) still alive; "
            f"skipping step={step}",
            flush=True,
        )
        return last_proc

    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    out_mp4 = os.path.join(samples_dir, f"step_{step}.mp4")
    log_path = os.path.join(samples_dir, f"step_{step}.log")

    cmd = [
        sys.executable,
        "-m",
        "examples.ReactiveGWM_casual_forcing.inference.sanity_sample",
        "--ckpt", ckpt_path,
        "--config", config_path,
        "--out", out_mp4,
        "--device", str(cfg.get("sanity_sample_device", "gpu_shared")),
    ]

    log_f = open(log_path, "w")  # closed when subprocess exits (kernel; log_f handle drops here)
    # Detach: new session so KeyboardInterrupt in trainer doesn't kill the child.
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(
        f"[Sanity-Spawn] pid={proc.pid} step={step} → {out_mp4} (log: {log_path})",
        flush=True,
    )
    return proc
