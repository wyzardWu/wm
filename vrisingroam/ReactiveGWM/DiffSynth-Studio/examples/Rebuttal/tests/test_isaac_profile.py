from __future__ import annotations

import unittest

from examples.Rebuttal.isaac_profile import (
    ISAAC_ACTION_COLUMNS,
    ISAAC_ACTION_INDICES,
    ISAAC_LATENT_FRAMES,
    ISAAC_PROFILE,
    ISAAC_RAW_FRAMES,
)
from examples.Rebuttal.variants import (
    FORMAL_NUM_PROCESSES,
    ISAAC_FORMAL_DEFAULTS,
    resolve_variant,
)


class IsaacProfileTests(unittest.TestCase):
    def test_locked_profile(self):
        self.assertEqual(ISAAC_PROFILE.num_buttons, 8)
        self.assertEqual(ISAAC_PROFILE.button_cols, ISAAC_ACTION_COLUMNS)
        self.assertEqual(ISAAC_PROFILE.default_width, 832)
        self.assertEqual(ISAAC_PROFILE.default_action_hold_window, 1)

    def test_causal_first_4n1_indices(self):
        self.assertEqual(ISAAC_RAW_FRAMES, 101)
        self.assertEqual(ISAAC_LATENT_FRAMES, 26)
        self.assertEqual(len(ISAAC_ACTION_INDICES), 26)
        self.assertEqual(ISAAC_ACTION_INDICES[:4], (0, 1, 5, 9))
        self.assertEqual(ISAAC_ACTION_INDICES[-1], 97)

    def test_variant_is_isolated(self):
        spec = resolve_variant("isaac_vanilla_strategy")
        self.assertEqual(spec.key, "isaac_v1")
        self.assertTrue(spec.is_isaac)
        self.assertEqual(FORMAL_NUM_PROCESSES["isaac_v1"], 4)
        self.assertEqual(ISAAC_FORMAL_DEFAULTS["width"], 832)
        self.assertEqual(ISAAC_FORMAL_DEFAULTS["action_hold_window"], 1)


if __name__ == "__main__":
    unittest.main()
