"""Action-conditioned T2V strategy for LTX-2.3 (ABot keys, per-frame context).

Wraps TextToVideoStrategy: before delegating to the stock prepare_training_inputs,
it rebuilds batch["conditions"]["video_prompt_embeds"] as the FLATTENED per-frame
context  [B, F * (L_cap + 64), D]:

    frame k ctx = [ caption (1024, projected pipeline output, shared) |
                    move_table[move_id_k] (32) | view_table[view_id_k] (32) ]

The stock strategy then treats it as an ordinary (long) prompt; the frame-fold
block patch (frame_context_patch.py) recognizes the F*L length and isolates
frames inside attn2. prompt_attention_mask is replaced with all-ones of the
new length. action_ids come from the conditions .pt sidecar field written by
prepare_conditions_actions.py.
"""
from __future__ import annotations

import torch

from ltx_trainer.training_strategies.flexible import FlexibleStrategy

ACT_LEN = 64          # move 32 + view 32
NUM_FRAMES = 16       # LTX latent frames for 121 pixel frames


class ActionT2VStrategy(FlexibleStrategy):
    def __init__(self, config, tables_path: str):
        super().__init__(config)
        blob = torch.load(tables_path, map_location="cpu", weights_only=True)
        self.move_table = blob["move_table"].float()   # [9, 32, D]
        self.view_table = blob["view_table"].float()
        assert self.move_table.shape[1] + self.view_table.shape[1] == ACT_LEN

    def prepare_training_inputs(self, batch, timestep_sampler):
        cond = batch["conditions"]
        cap = cond["video_prompt_embeds"]              # [B, 1024, D]
        ids = cond["action_ids"].long()                # [B, 16, 2]
        B, Lc, D = cap.shape
        F = ids.shape[1]
        assert F == NUM_FRAMES, f"expected {NUM_FRAMES} latent frames, got {F}"
        mt = self.move_table.to(cap.device, cap.dtype)
        vt = self.view_table.to(cap.device, cap.dtype)
        act = torch.cat([mt[ids[..., 0]], vt[ids[..., 1]]], dim=2)   # [B, F, 64, D]
        capf = cap.unsqueeze(1).expand(B, F, Lc, D)                  # shared caption
        ctx = torch.cat([capf, act], dim=2)                          # [B, F, Lc+64, D]
        cond["video_prompt_embeds"] = ctx.reshape(B, F * (Lc + ACT_LEN), D)
        cond["prompt_attention_mask"] = torch.ones(
            B, F * (Lc + ACT_LEN), dtype=torch.int64, device=cap.device
        )
        return super().prepare_training_inputs(batch, timestep_sampler)
