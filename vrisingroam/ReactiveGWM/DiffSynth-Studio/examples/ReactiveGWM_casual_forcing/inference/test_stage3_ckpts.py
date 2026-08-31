"""Stage 3 DMD ckpt 离线批量 sanity 测试 (一次性, 跑完即止).

为什么需要它
------------
Stage 3 训练中的 in-training sanity 推理在训练机上全部 **CUDA OOM** —— 8 卡训练把
GPU 0 占满 (训练进程 ~99GB + 邻居 ~21GB), spawn 出来的 sanity subprocess 抢不到显存,
连 T5 都加载不下. 结果 `stage3_dmd/samples/` 里只有 `step_*.log` (全是 OOM traceback),
**一个 mp4 都没产出**.

本脚本在**一张空闲 GPU** 上离线复跑, 对 `stage3_dmd/` 里现有的 `step-N.safetensors`
(EMA generator 导出, strict-loadable: 855 keys / missing=0) 批量出片, 供肉眼检查
4-step 因果生成质量 (动作响应 + 长程一致性 + 是否崩溃).

它**不重写推理逻辑**: 每个 (ckpt × 动作) 组合都以子进程调用已验证的
`inference/sanity_sample.py` (4-step KV-cache CFG rollout, 与训练 spawn 的同一份代码),
只负责: ① 挑空闲 GPU ② 枚举 ckpt × fill_action ③ 串行调度 + 收集日志 ④ 汇总.

参考 `stage1_ar/samples/` 的产物风格: 每个 ckpt 出 idle (`step_N.mp4`) +
light_punch (`step_N_light_punch.mp4`) 两个 60s 视频 (300 latent / 1500 像素帧 @ 25 FPS).

用法
----
    cd <repo root: .../DiffSynth-Studio>
    conda activate diffsynth_sf
    python -m examples.ReactiveGWM_casual_forcing.inference.test_stage3_ckpts \
        [--steps auto|latest|400,600,800] [--gpu auto|2] \
        [--fill_action idle,light_punch] [--seed 42]

或用便捷启动脚本 (自动 conda activate + cd repo root):
    bash examples/ReactiveGWM_casual_forcing/launch/test_stage3.sh [同上参数]

默认行为: 自动检测 stage3_dmd/ 下所有 step-N.safetensors, 每个出 idle+light_punch
两个视频, 固定 seed=42 (跨 ckpt 公平对比 —— 噪声相同, 差异纯来自模型).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# examples/ReactiveGWM_casual_forcing/inference/test_stage3_ckpts.py
#   parents[0]=inference  [1]=ReactiveGWM_casual_forcing  [2]=examples  [3]=DiffSynth-Studio
_HERE = Path(__file__).resolve()
_EXAMPLE_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[3]

_DEFAULT_CKPT_DIR = "/home/zeqingwang/zeqingwang/models/train/sf3_casual_forcing/stage3_dmd"
_DEFAULT_CONFIG = str(_EXAMPLE_DIR / "configs" / "stage3_dmd.yaml")
_STEP_RE = re.compile(r"^step-(\d+)\.safetensors$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3 DMD ckpt 离线批量 sanity 测试 (一次性)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt_dir", default=_DEFAULT_CKPT_DIR,
                   help="Stage 3 训练输出目录 (含 step-N.safetensors)")
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help="stage3 yaml (含 sanity_sample_diffusion_steps=4 等); 会与同目录 default.yaml 合并")
    p.add_argument("--steps", default="auto",
                   help="'auto'=测目录内全部 step-N; 'latest'=只测最大那个; 或逗号列表如 '400,600,800'")
    p.add_argument("--fill_action", default="idle,light_punch",
                   help="逗号分隔; 每个 ckpt 对每种各出一个视频 (clip 真实动作开头, 之后填充). 取值 idle / light_punch")
    p.add_argument("--gpu", default="auto",
                   help="'auto'=用 nvidia-smi 挑剩余显存最多的卡; 或物理卡号如 '2'")
    p.add_argument("--seed", type=int, default=42,
                   help="固定 rollout 噪声种子 (跨 ckpt 公平对比); 设 -1 则不固定 (随机)")
    p.add_argument("--out_dir", default=None,
                   help="mp4/log 输出目录 (默认 <ckpt_dir>/samples)")
    p.add_argument("--device", default="gpu_shared", choices=["gpu_shared", "cpu"],
                   help="透传给 sanity_sample.py; cpu 极慢, 仅兜底")
    p.add_argument("--memory_fraction", type=float, default=None,
                   help="透传给 sanity_sample.py --memory_fraction (按总显存比例的进程上限); "
                        "默认 None=沿用 config. 共享训练卡离线跑建议 0.35 (≈49GB), 否则 0.2 易 OOM")
    p.add_argument("--diffusion_steps", type=int, default=None,
                   help="透传给 sanity_sample.py --diffusion_steps (每帧去噪步数); "
                        "默认 None=沿用 config(Stage1/2=50). 提速可设 30/10, 整批时长近线性缩短")
    p.add_argument("--min_free_gb", type=float, default=40.0,
                   help="auto 挑卡时, 若最空闲的卡剩余显存低于此值则告警 (不强制中止)")
    p.add_argument("--retries", type=int, default=0,
                   help="单个视频失败时自动重试次数 (每次重新挑最空的卡 + 等 30s); 应对共享卡 OOM 竞态")
    p.add_argument("--skip_existing", action="store_true",
                   help="跳过 mp4 已存在且非空的 (step,fill) 组合 (watcher 增量模式用)")
    p.add_argument("--dry_run", action="store_true",
                   help="只打印将要执行的命令, 不真正推理")
    return p.parse_args()


def _pick_gpu(min_free_gb: float) -> int:
    """用 nvidia-smi 挑剩余显存 (MiB) 最多的物理卡号; 失败则回退 0."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:  # noqa: BLE001
        print(f"[test] nvidia-smi 不可用 ({e}); 回退 GPU 0", flush=True)
        return 0
    best_idx, best_free = 0, -1
    rows: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        idx_s, free_s = (x.strip() for x in line.split(","))
        idx, free = int(idx_s), int(free_s)
        rows.append((idx, free))
        if free > best_free:
            best_idx, best_free = idx, free
    summary = ", ".join(f"GPU{i}:{f/1024:.0f}GB" for i, f in rows)
    print(f"[test] GPU 剩余显存: {summary}", flush=True)
    print(f"[test] 自动选中 GPU {best_idx} (剩余 {best_free/1024:.1f}GB)", flush=True)
    if best_free / 1024.0 < min_free_gb:
        print(
            f"[test] ⚠️  GPU {best_idx} 剩余 {best_free/1024:.1f}GB < {min_free_gb}GB; "
            f"4-step CFG rollout 可能 OOM (可换卡或等空闲)",
            flush=True,
        )
    return best_idx


def _resolve_steps(ckpt_dir: Path, steps_arg: str) -> list[int]:
    """把 --steps 解析成排好序的 step 整数列表 (并校验对应 .safetensors 存在)."""
    present: dict[int, Path] = {}
    for f in ckpt_dir.iterdir():
        m = _STEP_RE.match(f.name)
        if m:
            present[int(m.group(1))] = f
    if not present:
        raise SystemExit(f"[test] {ckpt_dir} 下没有 step-N.safetensors")

    if steps_arg in ("auto", "all"):
        return sorted(present)
    if steps_arg == "latest":
        return [max(present)]
    # comma list
    want = [int(s) for s in steps_arg.split(",") if s.strip()]
    missing = [s for s in want if s not in present]
    if missing:
        avail = ", ".join(str(s) for s in sorted(present))
        raise SystemExit(f"[test] 指定的 step {missing} 不存在; 现有: {avail}")
    return sorted(want)


def _out_name(step: int, fill: str) -> str:
    """idle → step_N.mp4 (对齐 stage1 命名); 其余 → step_N_<fill>.mp4."""
    return f"step_{step}.mp4" if fill == "idle" else f"step_{step}_{fill}.mp4"


def _run_one(
    *, step: int, fill: str, ckpt: Path, out_mp4: Path, log_path: Path,
    config: str, device: str, seed: int, gpu: int, dry_run: bool,
    memory_fraction: float | None = None, diffusion_steps: int | None = None,
) -> tuple[bool, str]:
    """串行跑单个 (ckpt × fill) sanity_sample 子进程; 返回 (是否成功, 备注)."""
    cmd = [
        sys.executable, "-m",
        "examples.ReactiveGWM_casual_forcing.inference.sanity_sample",
        "--ckpt", str(ckpt),
        "--config", config,
        "--out", str(out_mp4),
        "--device", device,
        "--fill_action", fill,
    ]
    if seed >= 0:
        cmd += ["--seed", str(seed)]
    if memory_fraction is not None:
        cmd += ["--memory_fraction", str(memory_fraction)]
    if diffusion_steps is not None:
        cmd += ["--diffusion_steps", str(diffusion_steps)]

    env = os.environ.copy()
    if device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # sanity_sample 内 setdefault, 不会覆盖
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n[test] === step-{step} / {fill} → {out_mp4.name} (GPU {gpu}) ===", flush=True)
    if dry_run:
        print("[test] (dry-run) cmd: CUDA_VISIBLE_DEVICES=%s %s" % (
            env.get("CUDA_VISIBLE_DEVICES", "-"), " ".join(cmd)), flush=True)
        return True, "dry-run"

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env,
                              stdout=log_f, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    ok = proc.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0
    # 回看日志末尾, 把关键行 (写盘 / 报错) 打到控制台
    tail = _log_tail(log_path)
    if ok:
        print(f"[test] ✅ DONE ({dt:.0f}s)  {tail}", flush=True)
        return True, f"{dt:.0f}s"
    print(f"[test] ❌ FAIL (rc={proc.returncode}, {dt:.0f}s) — 详见 {log_path}", flush=True)
    print(f"[test]    {tail}", flush=True)
    return False, f"rc={proc.returncode}"


def _log_tail(log_path: Path, n: int = 6) -> str:
    """取日志里最有信息量的一行 (优先 '[Sanity] wrote'/error), 给控制台看个大概."""
    try:
        lines = log_path.read_text(errors="replace").strip().splitlines()
    except OSError:
        return ""
    for ln in reversed(lines):
        if "[Sanity] wrote" in ln:
            return ln.strip()
    for ln in reversed(lines):
        if "Error" in ln or "error" in ln or "Traceback" in ln:
            return ln.strip()
    return lines[-1].strip() if lines else ""


def _resolve_gpu(args: argparse.Namespace) -> int:
    """解析本次尝试用哪张卡: cpu→0 占位; 指定→该号; auto→当下最空的卡 (每次尝试重挑)."""
    if args.device == "cpu":
        return 0
    if args.gpu not in ("auto", ""):
        return int(args.gpu)
    return _pick_gpu(args.min_free_gb)


def _run_with_retries(
    *, step: int, fill: str, ckpt: Path, out_mp4: Path, log_path: Path,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    """跑单个视频, 失败时按 --retries 自动重试 (每次重新挑最空的卡 + 等 30s 让尖峰过去)."""
    attempts = max(1, args.retries + 1)
    note = ""
    for k in range(attempts):
        gpu = _resolve_gpu(args)
        ok, note = _run_one(
            step=step, fill=fill, ckpt=ckpt, out_mp4=out_mp4, log_path=log_path,
            config=args.config, device=args.device, seed=args.seed, gpu=gpu,
            dry_run=args.dry_run, memory_fraction=args.memory_fraction,
            diffusion_steps=args.diffusion_steps,
        )
        if ok:
            return True, (note if k == 0 else f"{note} (重试{k}次后成功)")
        if k + 1 < attempts:
            print(f"[test] ↻ step-{step}/{fill} 失败 ({note}); 30s 后换卡重试 ({k+1}/{args.retries})...",
                  flush=True)
            time.sleep(30)
    return False, f"{note} (重试{args.retries}次仍失败)"


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"[test] ckpt_dir 不存在: {ckpt_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = _resolve_steps(ckpt_dir, args.steps)
    fills = [f.strip() for f in args.fill_action.split(",") if f.strip()]
    gpu_mode = (
        ("auto-pick" if args.gpu in ("auto", "") else f"GPU {args.gpu}")
        if args.device != "cpu" else "cpu"
    )

    print(f"[test] ckpt_dir   = {ckpt_dir}", flush=True)
    print(f"[test] config     = {args.config}", flush=True)
    print(f"[test] steps      = {steps}", flush=True)
    print(f"[test] fill_action= {fills}", flush=True)
    print(f"[test] out_dir    = {out_dir}", flush=True)
    print(f"[test] device     = {args.device} ({gpu_mode})", flush=True)
    print(f"[test] mem_frac   = {args.memory_fraction if args.memory_fraction is not None else 'cfg默认'}", flush=True)
    print(f"[test] diff_steps = {args.diffusion_steps if args.diffusion_steps is not None else 'cfg默认(50)'}", flush=True)
    print(f"[test] seed       = {args.seed if args.seed >= 0 else 'random'}", flush=True)
    print(f"[test] retries    = {args.retries}  skip_existing = {args.skip_existing}", flush=True)
    print(f"[test] 共 {len(steps)}×{len(fills)} = {len(steps)*len(fills)} 个候选, 串行执行", flush=True)

    results: list[tuple[int, str, bool, str, Path]] = []
    for step in steps:
        ckpt = ckpt_dir / f"step-{step}.safetensors"
        for fill in fills:
            out_mp4 = out_dir / _out_name(step, fill)
            log_path = out_mp4.with_suffix(".log")
            if args.skip_existing and out_mp4.exists() and out_mp4.stat().st_size > 0:
                print(f"[test] ⏭  跳过 step-{step}/{fill}: {out_mp4.name} 已存在", flush=True)
                results.append((step, fill, True, "已存在(跳过)", out_mp4))
                continue
            if not ckpt.exists():  # keep_last_ckpts 可能在批测途中删掉旧 ckpt
                print(f"[test] ⚠️  跳过 step-{step}/{fill}: {ckpt.name} 已不存在 (被训练轮换删除?)",
                      flush=True)
                results.append((step, fill, False, "ckpt 消失", out_mp4))
                continue
            ok, note = _run_with_retries(
                step=step, fill=fill, ckpt=ckpt, out_mp4=out_mp4, log_path=log_path, args=args,
            )
            results.append((step, fill, ok, note, out_mp4))

    # 汇总
    print("\n" + "=" * 64, flush=True)
    print("[test] 汇总:", flush=True)
    n_ok = 0
    for step, fill, ok, note, out_mp4 in results:
        mark = "✅" if ok else "❌"
        n_ok += int(ok)
        loc = str(out_mp4) if ok else note
        print(f"  {mark} step-{step:<6} {fill:<12} {loc}", flush=True)
    print(f"[test] {n_ok}/{len(results)} 成功", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
