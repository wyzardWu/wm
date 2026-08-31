from __future__ import annotations

import re
import unittest

import torch
from peft import LoraConfig, inject_adapter_in_model

from examples.Rebuttal.trainable_policy import (
    CROSS_ATTN_LORA_TARGET,
    apply_full_dit_policy,
    apply_hybrid_cross_lora_policy,
    audit_full_dit_policy,
    audit_gradient_contract,
    audit_hybrid_cross_lora_policy,
    build_and_audit_optimizer,
)


class ToyAttention(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.o = torch.nn.Linear(dim, dim)
        self.norm_q = torch.nn.LayerNorm(dim)
        self.norm_k = torch.nn.LayerNorm(dim)


class ToyBlock(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = ToyAttention(dim)
        self.cross_attn = ToyAttention(dim)
        self.norm3 = torch.nn.LayerNorm(dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, 2 * dim),
            torch.nn.GELU(),
            torch.nn.Linear(2 * dim, dim),
        )
        self.modulation = torch.nn.Parameter(torch.zeros(1, 6, dim))


class ToyDiT(torch.nn.Module):
    def __init__(self, blocks: int = 2, dim: int = 8, buttons: int = 10):
        super().__init__()
        self.patch_embedding = torch.nn.Linear(dim, dim)
        self.blocks = torch.nn.ModuleList([ToyBlock(dim) for _ in range(blocks)])
        self.action_embedders = torch.nn.ModuleList(
            [torch.nn.Linear(buttons, dim, bias=False) for _ in range(blocks)]
        )
        self.head = torch.nn.Linear(dim, dim)


class FullPolicyTests(unittest.TestCase):
    def test_full_policy_trains_every_parameter(self):
        model = ToyDiT()
        model.requires_grad_(False)
        apply_full_dit_policy(model)
        audit = audit_full_dit_policy(model)
        self.assertEqual(audit.trainable, audit.total)
        self.assertEqual(audit.frozen.tensors, 0)
        self.assertEqual(audit.lora_tensors, 0)

    def test_uniform_optimizer_covers_every_trainable_once(self):
        model = ToyDiT()
        apply_full_dit_policy(model)
        optimizer = build_and_audit_optimizer(
            model,
            learning_rate=5e-5,
            weight_decay=0.01,
        )
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 5e-5)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)

    def test_full_gradient_contract(self):
        model = ToyDiT()
        apply_full_dit_policy(model)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        audit = audit_gradient_contract(model, mode="full_dit", step=1)
        self.assertEqual(
            set(audit.representatives),
            {"action", "self_attn", "cross_attn", "ffn"},
        )


class HybridPolicyTests(unittest.TestCase):
    def _model(self):
        blocks = 2
        dim = 8
        rank = 2
        model = ToyDiT(blocks=blocks, dim=dim)
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=CROSS_ATTN_LORA_TARGET,
        )
        model = inject_adapter_in_model(config, model)
        return model, blocks, dim, rank

    def test_regex_matches_cross_attention_only(self):
        target = re.compile(CROSS_ATTN_LORA_TARGET)
        self.assertIsNotNone(target.fullmatch("blocks.0.cross_attn.q"))
        self.assertIsNotNone(target.fullmatch("blocks.29.cross_attn.o"))
        self.assertIsNone(target.fullmatch("blocks.0.self_attn.q"))
        self.assertIsNone(target.fullmatch("blocks.0.cross_attn.norm_q"))

    def test_hybrid_policy_exact_freeze_contract(self):
        model, blocks, dim, rank = self._model()
        apply_hybrid_cross_lora_policy(model)
        expected_modules = blocks * 4
        expected_tensors = expected_modules * 2
        expected_scalars = expected_modules * rank * (dim + dim)
        audit = audit_hybrid_cross_lora_policy(
            model,
            expected_lora_modules=expected_modules,
            expected_lora_tensors=expected_tensors,
            expected_lora_scalars=expected_scalars,
        )
        self.assertEqual(audit.lora_modules, expected_modules)
        self.assertEqual(audit.lora_tensors, expected_tensors)
        self.assertEqual(audit.lora_scalars, expected_scalars)

        named = dict(model.named_parameters())
        self.assertTrue(
            named["blocks.0.cross_attn.q.lora_A.default.weight"].requires_grad
        )
        self.assertTrue(
            named["blocks.0.cross_attn.q.lora_B.default.weight"].requires_grad
        )
        self.assertFalse(named["blocks.0.cross_attn.q.base_layer.weight"].requires_grad)
        self.assertFalse(named["blocks.0.cross_attn.norm_q.weight"].requires_grad)
        self.assertFalse(named["blocks.0.norm3.weight"].requires_grad)
        self.assertTrue(named["blocks.0.self_attn.q.weight"].requires_grad)
        self.assertTrue(named["blocks.0.ffn.0.weight"].requires_grad)
        self.assertTrue(named["action_embedders.0.weight"].requires_grad)

    def test_policy_audit_detects_accidental_cross_unfreeze(self):
        model, blocks, dim, rank = self._model()
        apply_hybrid_cross_lora_policy(model)
        dict(model.named_parameters())[
            "blocks.0.cross_attn.q.base_layer.weight"
        ].requires_grad = True
        with self.assertRaisesRegex(AssertionError, "Hybrid policy mismatch"):
            audit_hybrid_cross_lora_policy(
                model,
                expected_lora_modules=blocks * 4,
                expected_lora_tensors=blocks * 8,
                expected_lora_scalars=blocks * 4 * rank * (dim + dim),
            )

    def test_uniform_optimizer_excludes_frozen_cross_branch(self):
        model, _, _, _ = self._model()
        apply_hybrid_cross_lora_policy(model)
        optimizer = build_and_audit_optimizer(
            model,
            learning_rate=5e-5,
            weight_decay=0.01,
        )
        optimizer_ids = {
            id(parameter) for parameter in optimizer.param_groups[0]["params"]
        }
        for name, parameter in model.named_parameters():
            if ".cross_attn." in name and ".lora_" not in name:
                self.assertNotIn(id(parameter), optimizer_ids, name)
            if ".norm3." in name:
                self.assertNotIn(id(parameter), optimizer_ids, name)

    def test_hybrid_gradient_contract_and_frozen_guard(self):
        model, _, _, _ = self._model()
        apply_hybrid_cross_lora_policy(model)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        audit = audit_gradient_contract(
            model,
            mode="hybrid_cross_lora",
            step=2,
            require_lora_a=True,
        )
        self.assertEqual(
            set(audit.representatives),
            {"action", "self_attn", "ffn", "lora_A", "lora_B"},
        )

        frozen = dict(model.named_parameters())[
            "blocks.0.cross_attn.q.base_layer.weight"
        ]
        frozen.grad = torch.ones_like(frozen)
        with self.assertRaisesRegex(AssertionError, "Frozen parameters"):
            audit_gradient_contract(
                model,
                mode="hybrid_cross_lora",
                step=2,
                require_lora_a=True,
            )


if __name__ == "__main__":
    unittest.main()
