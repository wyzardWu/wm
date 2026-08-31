"""WildWorld 三槽动作 strategy(全 prompt 注入,Incantation 式实体槽)。

帧 k 上下文 = [caption 1024(共享) | player 32 | boss 32 | cam 32] = 1120,
展平 [B, 16*1120, D] 交给 stock 策略;frame-fold patch 按 F*L 隔离各帧。
ids 来自 conditions 的 ww_action_ids [16,3](ww_prepare_sidecar.py 注入)。
"""
from __future__ import annotations

import torch

from ltx_trainer.training_strategies.flexible import FlexibleStrategy

WW_ACT_LEN = 96       # player 32 + boss 32 + cam 32
NUM_FRAMES = 16


class WWActionStrategy(FlexibleStrategy):
    def __init__(self, config, tables_path: str):
        super().__init__(config)
        blob = torch.load(tables_path, map_location="cpu", weights_only=True)
        self.pt = blob["player_table"].float()
        self.bt = blob["boss_table"].float()
        self.ct = blob["cam_table"].float()
        assert self.pt.shape[1] + self.bt.shape[1] + self.ct.shape[1] == WW_ACT_LEN

    def prepare_training_inputs(self, batch, timestep_sampler):
        cond = batch["conditions"]
        cap = cond["video_prompt_embeds"]              # [B, 1024, D]
        ids = cond["ww_action_ids"].long()             # [B, 16, 3]
        B, Lc, D = cap.shape
        F = ids.shape[1]
        assert F == NUM_FRAMES
        pt = self.pt.to(cap.device, cap.dtype)
        bt = self.bt.to(cap.device, cap.dtype)
        ct = self.ct.to(cap.device, cap.dtype)
        act = torch.cat([pt[ids[..., 0]], bt[ids[..., 1]], ct[ids[..., 2]]], dim=2)
        capf = cap.unsqueeze(1).expand(B, F, Lc, D)
        ctx = torch.cat([capf, act], dim=2)            # [B, F, 1120, D]
        cond["video_prompt_embeds"] = ctx.reshape(B, F * (Lc + WW_ACT_LEN), D)
        cond["prompt_attention_mask"] = torch.ones(
            B, F * (Lc + WW_ACT_LEN), dtype=torch.int64, device=cap.device)
        return super().prepare_training_inputs(batch, timestep_sampler)
