from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from examples.Rebuttal.eval_isaac_action import (
    NO_ACTION,
    action_names,
    make_eval_action,
)
from examples.Rebuttal.isaac_profile import ISAAC_ACTION_COLUMNS


class IsaacActionEvalTests(unittest.TestCase):
    def test_catalog_is_noop_plus_locked_eight_actions(self):
        self.assertEqual(action_names(), (NO_ACTION, *ISAAC_ACTION_COLUMNS))

    def test_no_action_is_all_zero(self):
        with patch.object(torch, "zeros", wraps=torch.zeros) as zeros:
            action = make_eval_action(NO_ACTION, "cpu")
        self.assertEqual(tuple(action.shape), (1, 101, 8))
        self.assertEqual(action.dtype, torch.bfloat16)
        self.assertEqual(action.count_nonzero().item(), 0)
        zeros.assert_called_once()

    def test_each_action_is_one_hot_after_anchor_frame(self):
        for column, name in enumerate(ISAAC_ACTION_COLUMNS):
            with self.subTest(name=name):
                action = make_eval_action(name, "cpu")
                self.assertEqual(action[:, 0].count_nonzero().item(), 0)
                self.assertEqual(action[:, 1:, column].sum().item(), 100)
                action[:, 1:, column] = 0
                self.assertEqual(action.count_nonzero().item(), 0)

    def test_unknown_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown Isaac action"):
            make_eval_action("JUMP", "cpu")


if __name__ == "__main__":
    unittest.main()
