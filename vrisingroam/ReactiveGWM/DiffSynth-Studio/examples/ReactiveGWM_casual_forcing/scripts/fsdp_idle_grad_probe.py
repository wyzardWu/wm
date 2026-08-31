"""FSDP idle param-group grad 探针 —— 验证 Stage 3 单-optimizer 两-group 的核心假设.

背景: ``modules/runner_dmd.py`` 用 **单 AdamW + 两 param-group**(generator / critic)
承载不同 lr/betas, 靠 "非活跃 phase 的参数 grad 为 None → AdamW 不更新" 来复刻上游
"双独立 optimizer 交替更新"。该假设在 DDP / 单卡成立(图外参数 grad 天然 None),但
**FSDP 把 generator+critic 合进同一个 FlatParameter(NO_WRAP 单 unit)**, 非活跃组可能
拿到 grad=0(而非 None)→ 被 AdamW 每步 weight-decay + 二阶矩衰减(generator 有效 lr
约被放大 2.4×)。

本脚本用玩具模型(两个 Linear, 与真实结构同构)复刻该场景, 直接观测:
  1) generator-phase backward 后, idle(critic) 组的 orig-param grad 是 None 还是 zeros;
  2) 现状(不打补丁)下 optimizer.step 是否会给 critic 记 AdamW state(= 被误更新);
  3) 打补丁(step 前把 idle 组 grad 置 None, 即 runner 现已采用的修复)后 critic 是否被跳过.

判定:
  - "idle grad = None"  → 假设成立, 问题② 不存在(但 grad-clip 子集问题① 仍需 model.parameters() 修复);
  - "idle grad = zeros" → 问题② 确认: 不打补丁时 critic 会被 AdamW 误更新; 补丁后应跳过.

跑法(只需 1 张空闲卡; 玩具模型 <1GB, 几秒):
    CUDA_VISIBLE_DEVICES=<free_gpu> torchrun --nproc_per_node=1 \
        examples/ReactiveGWM_casual_forcing/scripts/fsdp_idle_grad_probe.py
  或直接(单进程, 脚本会补齐 dist 环境变量):
    CUDA_VISIBLE_DEVICES=<free_gpu> python examples/ReactiveGWM_casual_forcing/scripts/fsdp_idle_grad_probe.py
  最faithful是真分片(需 2 张空闲卡):
    CUDA_VISIBLE_DEVICES=<g1>,<g2> torchrun --nproc_per_node=2 .../fsdp_idle_grad_probe.py

注意: world_size=1 时 FULL_SHARD 不真正跨卡分片, 但 FlatParameter 的 grad-view 逻辑
(`_use_sharded_grad_views` / `_is_grad_none_mask`)与 >1 卡相同 → 足以判定 None vs 0.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


class ToyModule(nn.Module):
    """两组可训练参数, 同 dtype/requires_grad → 进同一个 FlatParameter(单 unit)."""

    def __init__(self) -> None:
        super().__init__()
        self.gen = nn.Linear(16, 16, bias=False)   # "generator"
        self.crit = nn.Linear(16, 16, bias=False)  # "critic"

    def forward(self, x: torch.Tensor, phase: str) -> torch.Tensor:
        if phase == "gen":
            # generator phase: 只有 gen 进图; crit 在 no_grad 下被用到(像 rollout 里的
            # no_grad generator), 即 critic 作为 idle 组.
            with torch.no_grad():
                _ = self.crit(x)
            return self.gen(x).square().mean()
        # critic phase: 只有 crit 进图; gen 在 no_grad 下被用到(idle 组).
        with torch.no_grad():
            h = self.gen(x)
        return self.crit(h).square().mean()


def _params_by_name(model: nn.Module, key: str) -> list[torch.nn.Parameter]:
    """按名字挑出 gen / crit 的 orig 参数(use_orig_params=True 下名字保留)."""
    return [p for n, p in model.named_parameters() if key in n and p.requires_grad]


def _grad_state(model: nn.Module, key: str, rank: int, tag: str) -> None:
    ps = _params_by_name(model, key)
    if not ps:
        if rank == 0:
            print(f"[{tag}] ({key} 在本 rank 视图里无参数)", flush=True)
        return
    g = ps[0].grad
    if g is None:
        msg = "None"
    else:
        is_zero = bool((g.abs().max() == 0).item())
        msg = f"zeros={is_zero}  numel={g.numel()}  absmax={g.abs().max().item():.3e}"
    if rank == 0:
        print(f"[{tag}] idle({key}) orig-param grad = {msg}", flush=True)


def main() -> None:
    # 允许不经 torchrun 直接 `python` 单进程跑.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not torch.cuda.is_available():
        raise SystemExit("本探针需要 CUDA(FSDP 在 torch 2.x 不支持纯 CPU)。请在有空闲卡时跑。")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(0)

    if rank == 0:
        print(f"=== FSDP idle-grad 探针: world_size={world} (FULL_SHARD, use_orig_params=True) ===", flush=True)

    model = ToyModule().to(dev)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,   # 与 launch --fsdp_use_orig_params true 一致
        device_id=local,
    )

    opt = torch.optim.AdamW(
        [
            {"params": _params_by_name(model, "gen"),  "lr": 2.0e-6, "betas": (0.0, 0.999), "weight_decay": 0.01},
            {"params": _params_by_name(model, "crit"), "lr": 4.0e-7, "betas": (0.0, 0.999), "weight_decay": 0.01},
        ]
    )
    x = torch.randn(4, 16, device=dev)

    def crit_step_count() -> list[int]:
        return [
            int(opt.state[p]["step"])
            for p in _params_by_name(model, "crit")
            if p in opt.state and "step" in opt.state[p]
        ]

    # ---------- A) 现状: generator-phase, 不打补丁 ----------
    opt.zero_grad(set_to_none=True)
    loss = model(x, "gen")
    loss.backward()
    _grad_state(model, "crit", rank, "A. gen-phase backward(现状)")
    opt.step()
    if rank == 0:
        sc = crit_step_count()
        verdict = ">0 → idle 组被 AdamW 误更新 ⇒ 问题② 成立" if (sc and sc[0] > 0) else "空/0 → 未被更新(假设成立)"
        print(f"[A. 现状] step 后 critic 的 AdamW step 计数 = {sc}  ({verdict})", flush=True)

    # ---------- B) 打补丁: step 前把 idle(critic) 组置 None ----------
    # 清掉 A 留下的 critic optimizer state, 干净对比.
    for p in _params_by_name(model, "crit"):
        opt.state.pop(p, None)
    opt.zero_grad(set_to_none=True)
    loss = model(x, "gen")
    loss.backward()
    for p in _params_by_name(model, "crit"):   # ← runner 现已采用的修复
        p.grad = None
    _grad_state(model, "crit", rank, "B. gen-phase backward + 置None(修复后)")
    opt.step()
    if rank == 0:
        sc = crit_step_count()
        verdict = "空/0 → idle 组被正确跳过 ✓" if not sc else f"{sc} → 仍被更新(异常!)"
        print(f"[B. 修复后] step 后 critic 的 AdamW step 计数 = {sc}  ({verdict})", flush=True)
        print("=== 结论 ===", flush=True)
        print("若 A 显示 grad=zeros 且 step>0 → 问题② 在本 FSDP 版本成立, 修复(B)生效; "
              "若 A 显示 grad=None → 假设本就成立, 修复为无害冗余。无论哪种, model.parameters() "
              "传全集修复了 grad-clip 的问题①。", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
