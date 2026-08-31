from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file

from examples.Rebuttal.checkpoint_io import (
    CHECKPOINT_SCHEMA_VERSION,
    RebuttalCheckpointLogger,
    atomic_write_json,
    fuse_hybrid_checkpoint_into_model,
    load_and_validate_hybrid_manifest,
    save_safetensors_atomic,
    sha256_file,
    split_hybrid_state_dict,
    validate_full_dit_checkpoint,
    validate_hybrid_checkpoint_parts,
)
from examples.Rebuttal.trainable_policy import (
    apply_full_dit_policy,
    apply_hybrid_cross_lora_policy,
)
from examples.Rebuttal.variants import (
    CROSS_ATTN_LORA_TARGET,
    LORA_ALPHA,
    LORA_RANK,
)


def hybrid_parts(rank: int = 2):
    full = {
        "action_embedders.0.weight": torch.randn(4, 2),
        "blocks.0.self_attn.q.weight": torch.randn(4, 4),
        "blocks.0.ffn.0.weight": torch.randn(8, 4),
    }
    lora = {
        "blocks.0.cross_attn.q.lora_A.default.weight": torch.randn(rank, 4),
        "blocks.0.cross_attn.q.lora_B.default.weight": torch.randn(4, rank),
    }
    return full, lora


class CheckpointContractTests(unittest.TestCase):
    def test_split_hybrid_state(self):
        full, lora = hybrid_parts()
        combined = {**full, **lora}
        got_full, got_lora = split_hybrid_state_dict(combined)
        self.assertEqual(set(got_full), set(full))
        self.assertEqual(set(got_lora), set(lora))

    def test_rejects_frozen_cross_weight_in_full_delta(self):
        full, lora = hybrid_parts()
        full["blocks.0.cross_attn.q.base_layer.weight"] = torch.randn(4, 4)
        with self.assertRaisesRegex(ValueError, "frozen cross-branch"):
            validate_hybrid_checkpoint_parts(full, lora)

    def test_rejects_norm3_in_full_delta(self):
        full, lora = hybrid_parts()
        full["blocks.0.norm3.weight"] = torch.randn(4)
        with self.assertRaisesRegex(ValueError, "frozen cross-branch"):
            validate_hybrid_checkpoint_parts(full, lora)

    def test_rejects_self_attention_lora(self):
        full, _ = hybrid_parts()
        wrong = {
            "blocks.0.self_attn.q.lora_A.default.weight": torch.randn(2, 4),
            "blocks.0.self_attn.q.lora_B.default.weight": torch.randn(4, 2),
        }
        with self.assertRaisesRegex(ValueError, "non-cross-q/k/v/o"):
            validate_hybrid_checkpoint_parts(full, wrong)

    def test_full_checkpoint_requires_all_major_components(self):
        state = {
            "action_embedders.0.weight": torch.randn(4, 2),
            "blocks.0.self_attn.q.weight": torch.randn(4, 4),
            "blocks.0.cross_attn.q.weight": torch.randn(4, 4),
            "blocks.0.ffn.0.weight": torch.randn(8, 4),
        }
        validate_full_dit_checkpoint(state)
        del state["blocks.0.cross_attn.q.weight"]
        with self.assertRaisesRegex(ValueError, "cross_attn"):
            validate_full_dit_checkpoint(state)


class FuseTests(unittest.TestCase):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_embedders = torch.nn.ModuleList(
                [torch.nn.Linear(2, 4, bias=False)]
            )
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.q = torch.nn.Linear(4, 4)
            block.cross_attn = torch.nn.Module()
            block.cross_attn.q = torch.nn.Linear(4, 4, bias=False)
            block.ffn = torch.nn.Sequential(torch.nn.Linear(4, 8))
            self.blocks = torch.nn.ModuleList([block])

    def test_fuse_matches_alpha_over_rank_formula(self):
        torch.manual_seed(0)
        model = self.Model()
        original = model.blocks[0].cross_attn.q.weight.detach().clone()
        full, lora = hybrid_parts(rank=2)
        a = lora["blocks.0.cross_attn.q.lora_A.default.weight"]
        b = lora["blocks.0.cross_attn.q.lora_B.default.weight"]
        fused = fuse_hybrid_checkpoint_into_model(
            model,
            full_state=full,
            lora_state=lora,
            rank=2,
            alpha=4,
        )
        self.assertEqual(fused, 1)
        expected = original + 2 * torch.mm(b, a)
        torch.testing.assert_close(model.blocks[0].cross_attn.q.weight, expected)
        torch.testing.assert_close(
            model.action_embedders[0].weight,
            full["action_embedders.0.weight"],
        )

    def test_bfloat16_fuse_matches_inference_loader_order(self):
        torch.manual_seed(1)
        model = self.Model().to(torch.bfloat16)
        original = model.blocks[0].cross_attn.q.weight.detach().clone()
        full, lora = hybrid_parts(rank=2)
        lora = {name: tensor.to(torch.bfloat16) for name, tensor in lora.items()}
        a = lora["blocks.0.cross_attn.q.lora_A.default.weight"]
        b = lora["blocks.0.cross_attn.q.lora_B.default.weight"]
        expected = original + 2 * torch.mm(b, a)
        fuse_hybrid_checkpoint_into_model(
            model,
            full_state=full,
            lora_state=lora,
            rank=2,
            alpha=4,
        )
        torch.testing.assert_close(
            model.blocks[0].cross_attn.q.weight,
            expected,
            rtol=0,
            atol=0,
        )


class ManifestTests(unittest.TestCase):
    def test_manifest_hash_validation(self):
        full, lora = hybrid_parts(rank=LORA_RANK)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_path = root / "step-10.full.safetensors"
            lora_path = root / "step-10.lora.safetensors"
            manifest_path = root / "step-10.manifest.json"
            save_safetensors_atomic(full, full_path)
            save_safetensors_atomic(lora, lora_path)
            manifest = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "variant": "v3",
                "step": 10,
                "lora": {
                    "rank": LORA_RANK,
                    "alpha": LORA_ALPHA,
                    "target_modules": CROSS_ATTN_LORA_TARGET,
                },
                "files": {
                    "full": {
                        "name": full_path.name,
                        "sha256": sha256_file(full_path),
                    },
                    "lora": {
                        "name": lora_path.name,
                        "sha256": sha256_file(lora_path),
                    },
                },
            }
            atomic_write_json(manifest_path, manifest)
            loaded = load_and_validate_hybrid_manifest(manifest_path)
            self.assertEqual(loaded["step"], 10)

            tampered = json.loads(manifest_path.read_text())
            tampered["files"]["lora"]["sha256"] = "0" * 64
            atomic_write_json(manifest_path, tampered)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_and_validate_hybrid_manifest(manifest_path)


class TinyCheckpointDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_embedders = torch.nn.ModuleList([torch.nn.Linear(2, 4, bias=False)])
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        block.self_attn.q = torch.nn.Linear(4, 4)
        block.cross_attn = torch.nn.Module()
        block.cross_attn.q = torch.nn.Linear(4, 4)
        block.cross_attn.norm_q = torch.nn.LayerNorm(4)
        block.norm3 = torch.nn.LayerNorm(4)
        block.ffn = torch.nn.Sequential(torch.nn.Linear(4, 8))
        self.blocks = torch.nn.ModuleList([block])
        self.head = torch.nn.Linear(4, 4)


class TinyTrainingWrapper(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.pipe = torch.nn.Module()
        self.pipe.dit = dit

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        names = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        selected = {
            name: tensor for name, tensor in state_dict.items() if name in names
        }
        if remove_prefix:
            selected = {
                name.removeprefix(remove_prefix): tensor
                for name, tensor in selected.items()
            }
        return selected


class FakeSaveAccelerator:
    is_main_process = True

    def wait_for_everyone(self):
        return None

    def get_state_dict(self, model):
        return model.state_dict()

    def unwrap_model(self, model):
        return model

    def save(self, state_dict, path, safe_serialization=True):
        self.assert_safe = safe_serialization
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in state_dict.items()
            },
            path,
        )


class LoggerTests(unittest.TestCase):
    def _logger(self, root, variant):
        return RebuttalCheckpointLogger(
            root,
            variant=variant,
            base_model_fingerprint={"root": "raw-wan"},
            metadata={"sha256": "abc"},
            training_config={"learning_rate": 5e-5},
            parameter_audit={"mode": variant},
        )

    def test_v1_logger_writes_one_standard_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            dit = TinyCheckpointDiT()
            apply_full_dit_policy(dit)
            model = TinyTrainingWrapper(dit)
            logger = self._logger(directory, "v1")
            logger.save_model(FakeSaveAccelerator(), model, 5)
            self.assertTrue((Path(directory) / "step-5.safetensors").is_file())
            self.assertFalse((Path(directory) / "step-5.manifest.json").exists())

    def test_v3_logger_writes_split_files_and_valid_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            dit = TinyCheckpointDiT()
            dit = inject_adapter_in_model(
                LoraConfig(
                    r=LORA_RANK,
                    lora_alpha=LORA_ALPHA,
                    target_modules=CROSS_ATTN_LORA_TARGET,
                ),
                dit,
            )
            apply_hybrid_cross_lora_policy(dit)
            model = TinyTrainingWrapper(dit)
            logger = self._logger(directory, "v3")
            logger.save_model(FakeSaveAccelerator(), model, 5)
            manifest_path = Path(directory) / "step-5.manifest.json"
            manifest = load_and_validate_hybrid_manifest(manifest_path)
            self.assertEqual(manifest["step"], 5)
            self.assertTrue((Path(directory) / "step-5.full.safetensors").is_file())
            self.assertTrue((Path(directory) / "step-5.lora.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
