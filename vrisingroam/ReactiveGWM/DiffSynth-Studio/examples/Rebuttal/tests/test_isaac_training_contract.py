from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from examples.Rebuttal.preflight import (
    validate_formal_hyperparameters,
    validate_metadata,
)
from examples.Rebuttal.variants import ISAAC_FORMAL_DEFAULTS, VARIANTS


class IsaacTrainingContractTests(unittest.TestCase):
    def test_formal_recipe(self):
        values = dict(ISAAC_FORMAL_DEFAULTS)
        values["expected_num_processes"] = values.pop("num_processes")
        values["variant"] = "isaac_v1"
        args = SimpleNamespace(**values)
        validate_formal_hyperparameters(args)
        args.action_hold_window = 10
        with self.assertRaisesRegex(ValueError, "approved recipe"):
            validate_formal_hyperparameters(args)

    def test_expected_duplicates_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            rows = []
            for index in range(4_119):
                rows.append(
                    {
                        "video": f"video/{index}.mp4",
                        "action": f"action/{index}.parquet",
                        "prompt": f"vanilla strategy sentence {index}",
                    }
                )
            rows.extend(rows[:881])
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("video", "action", "prompt")
                )
                writer.writeheader()
                writer.writerows(rows)
            with mock.patch(
                "examples.Rebuttal.preflight.sha256_file", return_value="sha"
            ), mock.patch(
                "examples.Rebuttal.preflight.md5_file", return_value="md5"
            ):
                audit = validate_metadata(VARIANTS["isaac_v1"], path)
            self.assertEqual(audit.rows, 5_000)
            self.assertEqual(audit.unique_clips, 4_119)


if __name__ == "__main__":
    unittest.main()
