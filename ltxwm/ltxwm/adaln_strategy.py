"""AdaLN 臂的训练 strategy:caption 走原生全局 cross-attention(不折叠),
动作 ids 每个 microbatch 塞进 HOLDER 由 adaln wrapper 消费。
10% 样本级 dropout 置为 idle(0,0),给推理期动作 CFG 留口子。
"""
from __future__ import annotations

import torch

from ltx_trainer.training_strategies.flexible import FlexibleStrategy

from ltxwm.action_adaln_patch import HOLDER

NUM_FRAMES = 16


class AdaLNActionStrategy(FlexibleStrategy):
    def __init__(self, config, action_dropout: float = 0.1):
        super().__init__(config)
        self.action_dropout = action_dropout

    def prepare_training_inputs(self, batch, timestep_sampler):
        ids = batch["conditions"]["action_ids"].long()      # [B, 16, 2]
        assert ids.shape[1] == NUM_FRAMES, f"latent frames {ids.shape[1]} != {NUM_FRAMES}"
        if self.action_dropout > 0:
            drop = torch.rand(ids.shape[0], device=ids.device) < self.action_dropout
            ids = ids.clone()
            ids[drop] = 0
        HOLDER["ids"] = ids
        return super().prepare_training_inputs(batch, timestep_sampler)
