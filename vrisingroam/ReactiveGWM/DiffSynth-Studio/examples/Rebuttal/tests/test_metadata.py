from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REBUTTAL_DIR = Path(__file__).resolve().parents[1]
if str(REBUTTAL_DIR) not in sys.path:
    sys.path.insert(0, str(REBUTTAL_DIR))

from prepare_metadata import (  # noqa: E402
    MetadataValidationError,
    build_variant_frames,
    extract_strategy,
    generate_metadata,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExtractStrategyTests(unittest.TestCase):
    def test_extracts_terminal_strategy(self):
        prompt = (
            "NPC: Active_Behavior(N/A), Passive_Behavior(N/A), "
            "Strategy(Control: Keeps range.)"
        )
        self.assertEqual(
            extract_strategy(prompt),
            "Strategy(Control: Keeps range.)",
        )

    def test_allows_nested_parentheses(self):
        prompt = "NPC: Strategy(Control: Keeps range (when possible).)"
        self.assertEqual(
            extract_strategy(prompt),
            "Strategy(Control: Keeps range (when possible).)",
        )

    def test_rejects_trailing_text(self):
        with self.assertRaises(MetadataValidationError):
            extract_strategy("NPC: Strategy(Control: Keeps range.) trailing")

    def test_rejects_multiple_strategy_markers(self):
        with self.assertRaises(MetadataValidationError):
            extract_strategy("Strategy(A: one) Strategy(B: two)")


class BuildVariantTests(unittest.TestCase):
    def setUp(self):
        self.vanilla = pd.DataFrame(
            [
                {"video": "v1.mp4", "action": "a1.parquet", "prompt": "Narration one."},
                {
                    "video": "v2.mp4",
                    "action": "a2.parquet",
                    "prompt": "Narration two.  ",
                },
            ]
        )
        self.structured = pd.DataFrame(
            [
                {
                    "video": "v2.mp4",
                    "action": "a2.parquet",
                    "prompt": "NPC: Active_Behavior(N/A), Strategy(Defense: Hold.)",
                },
                {
                    "video": "v1.mp4",
                    "action": "a1.parquet",
                    "prompt": "NPC: Active_Behavior(N/A), Strategy(Offense: Push.)",
                },
            ]
        )

    def test_key_join_preserves_vanilla_order(self):
        v1, v2, strategies = build_variant_frames(self.vanilla, self.structured)
        self.assertEqual(v1["video"].tolist(), ["v1.mp4", "v2.mp4"])
        self.assertEqual(
            v1["prompt"].tolist(),
            [
                "Narration one. Strategy(Offense: Push.)",
                "Narration two. Strategy(Defense: Hold.)",
            ],
        )
        self.assertEqual(
            v2["prompt"].tolist(),
            ["Strategy(Offense: Push.)", "Strategy(Defense: Hold.)"],
        )
        self.assertEqual(strategies, v2["prompt"].tolist())

    def test_rejects_mismatched_keys(self):
        structured = self.structured.iloc[:1].copy()
        with self.assertRaises(MetadataValidationError):
            build_variant_frames(self.vanilla, structured)

    def test_generate_is_atomic_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vanilla_path = root / "vanilla.csv"
            structured_path = root / "structured.csv"
            output_dir = root / "out"
            self.vanilla.to_csv(vanilla_path, index=False)
            self.structured.to_csv(structured_path, index=False)
            before = (_sha256(vanilla_path), _sha256(structured_path))

            manifest = generate_metadata(
                vanilla_path,
                structured_path,
                output_dir,
                tokenizer_path=None,
                smoke_rows=1,
            )

            after = (_sha256(vanilla_path), _sha256(structured_path))
            self.assertEqual(before, after)
            self.assertEqual(manifest["outputs"]["v1"]["rows"], 2)
            self.assertEqual(manifest["outputs"]["v2"]["rows"], 2)
            manifest_disk = json.loads(
                (output_dir / "metadata_manifest.json").read_text()
            )
            self.assertEqual(
                manifest_disk["outputs"]["v1"]["sha256"],
                _sha256(output_dir / "metadata_v1_vanilla_strategy.csv"),
            )
            self.assertTrue((output_dir / "metadata_v1_smoke1.csv").is_file())
            self.assertTrue((output_dir / "metadata_v2_smoke1.csv").is_file())


if __name__ == "__main__":
    unittest.main()
