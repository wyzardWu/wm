from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from examples.Rebuttal.runner import launch_rebuttal_training


class TinyDataset(torch.utils.data.Dataset):
    load_from_cache = False

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"x": torch.tensor(float(index + 1))}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        return (self.weight * data["x"] - 0.5).square()


class RecordingLogger:
    def __init__(self, output_path, initial_step=0):
        self.output_path = str(output_path)
        self.num_steps = initial_step
        self.saved_at_end = False

    def on_step_end(self, accelerator, model, save_steps, **kwargs):
        self.num_steps += 1

    def on_training_end(self, accelerator, model, save_steps):
        self.saved_at_end = True


class FakeAccelerator:
    device = torch.device("cpu")
    is_main_process = False
    sync_gradients = True
    state = SimpleNamespace(deepspeed_plugin=None)

    def prepare(self, *objects):
        return objects

    def accumulate(self, model):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        return None


class RunnerTests(unittest.TestCase):
    def test_stops_at_exact_step_and_recycles_dataloader(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                max_train_steps=5,
                learning_rate=5e-5,
                weight_decay=0.01,
                dataset_num_workers=0,
                save_steps=1000,
                resume_state=None,
                save_full_state=False,
                allow_checkpoint_overwrite=False,
            )
            logger = RecordingLogger(Path(directory))
            result = launch_rebuttal_training(
                FakeAccelerator(),
                TinyDataset(),
                TinyModel(),
                logger,
                args=args,
            )
            self.assertEqual(result.initial_step, 0)
            self.assertEqual(result.final_step, 5)
            self.assertEqual(result.dataloader_epochs, 3)
            self.assertTrue(logger.saved_at_end)

    def test_resumed_counter_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                max_train_steps=5,
                learning_rate=5e-5,
                weight_decay=0.01,
                dataset_num_workers=0,
                save_steps=1000,
                resume_state=None,
                save_full_state=False,
                allow_checkpoint_overwrite=False,
            )
            logger = RecordingLogger(Path(directory), initial_step=3)
            result = launch_rebuttal_training(
                FakeAccelerator(),
                TinyDataset(),
                TinyModel(),
                logger,
                args=args,
            )
            self.assertEqual(result.initial_step, 3)
            self.assertEqual(result.final_step, 5)
            self.assertEqual(result.dataloader_epochs, 1)


if __name__ == "__main__":
    unittest.main()
