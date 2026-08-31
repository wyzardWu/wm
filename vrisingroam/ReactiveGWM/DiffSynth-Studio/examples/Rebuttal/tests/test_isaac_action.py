from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from examples.Rebuttal.isaac_model import IsaacReactiveGWMModel
from examples.Rebuttal.isaac_profile import (
    ISAAC_ACTION_COLUMNS,
    ISAAC_ACTION_INDICES,
    read_isaac_action,
)


class IsaacActionTests(unittest.TestCase):
    def test_strict_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.parquet"
            pd.DataFrame(
                {column: [index % 2] * 101 for index, column in enumerate(ISAAC_ACTION_COLUMNS)}
            ).to_parquet(path)
            action = read_isaac_action(path)
            self.assertEqual(tuple(action.shape), (101, 8))
            self.assertEqual(action.dtype, torch.float32)

    def test_rejects_wrong_length_and_non_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.parquet"
            frame = pd.DataFrame({column: [0] * 100 for column in ISAAC_ACTION_COLUMNS})
            frame.to_parquet(path)
            with self.assertRaisesRegex(ValueError, "exactly 101"):
                read_isaac_action(path)
            frame = pd.DataFrame({column: [0] * 101 for column in ISAAC_ACTION_COLUMNS})
            frame.loc[0, ISAAC_ACTION_COLUMNS[0]] = 2
            frame.to_parquet(path)
            with self.assertRaisesRegex(ValueError, "binary"):
                read_isaac_action(path)

    def test_model_mapping_without_constructing_5b_backbone(self):
        model = object.__new__(IsaacReactiveGWMModel)
        action = torch.arange(101 * 8).reshape(1, 101, 8)
        result = IsaacReactiveGWMModel.prepare_action_binned(model, action, 26)
        self.assertTrue(torch.equal(result, action[:, list(ISAAC_ACTION_INDICES)]))
        with self.assertRaisesRegex(ValueError, "shape"):
            IsaacReactiveGWMModel.prepare_action_binned(model, action[:, :-1], 26)
        with self.assertRaisesRegex(ValueError, "latent"):
            IsaacReactiveGWMModel.prepare_action_binned(model, action, 25)


if __name__ == "__main__":
    unittest.main()
