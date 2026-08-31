from __future__ import annotations

import unittest

from examples.Rebuttal.eval import replace_strategy, strategy_catalog


CONTROL = "Strategy(Control: replacement.)"


class EvalPromptTests(unittest.TestCase):
    def test_v1_replaces_only_terminal_strategy_and_keeps_one_space(self):
        prompt = "Vanilla narration Strategy(Defense: old.)"
        result = replace_strategy(prompt, CONTROL, "vanilla_strategy")
        self.assertEqual(result, "Vanilla narration " + CONTROL)

    def test_v2_is_strategy_only(self):
        result = replace_strategy(
            "Strategy(Defense: old.)",
            CONTROL,
            "strategy_only",
        )
        self.assertEqual(result, CONTROL)

    def test_v3_keeps_active_and_passive_byte_exact(self):
        prefix = "NPC: Active_Behavior(A), Passive_Behavior(B), "
        prompt = prefix + "Strategy(Defense: old.)"
        result = replace_strategy(prompt, CONTROL, "structured")
        self.assertEqual(result, prefix + CONTROL)

    def test_real_catalog_has_nine_strategies(self):
        self.assertEqual(len(strategy_catalog()), 9)


if __name__ == "__main__":
    unittest.main()
