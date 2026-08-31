from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from examples.Rebuttal.preflight import (
    md5_file,
    validate_cache,
    validate_formal_hyperparameters,
    validate_metadata,
)
from examples.Rebuttal.prepare_metadata import MetadataValidationError
from examples.Rebuttal.variants import (
    FORMAL_DEFAULTS,
    FORMAL_NUM_PROCESSES,
    VARIANTS,
)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("video", "action", "prompt"))
        writer.writeheader()
        writer.writerows(rows)


class MetadataPreflightTests(unittest.TestCase):
    def test_real_metadata_contracts(self):
        for spec in VARIANTS.values():
            with self.subTest(variant=spec.key):
                audit = validate_metadata(spec, spec.metadata_path)
                self.assertEqual(audit.rows, 5_000 if spec.is_isaac else 10_000)
                if spec.is_isaac:
                    self.assertEqual(audit.unique_clips, 4_119)
                else:
                    self.assertTrue(audit.exact_manifest_match)

    def test_v1_rejects_newline_separator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            write_csv(
                path,
                [
                    {
                        "video": "x.mp4",
                        "action": "x.parquet",
                        "prompt": "Vanilla text\nStrategy(Control: test.)",
                    }
                ],
            )
            with self.assertRaisesRegex(MetadataValidationError, "one ASCII space"):
                validate_metadata(VARIANTS["v1"], path)

    def test_v2_rejects_behavior_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            write_csv(
                path,
                [
                    {
                        "video": "x.mp4",
                        "action": "x.parquet",
                        "prompt": "Active_Behavior(X) Strategy(Control: test.)",
                    }
                ],
            )
            with self.assertRaisesRegex(MetadataValidationError, "exactly"):
                validate_metadata(VARIANTS["v2"], path)


class CachePreflightTests(unittest.TestCase):
    def test_cache_contract_and_shared_vae_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.csv"
            write_csv(
                metadata_path,
                [
                    {
                        "video": "x.mp4",
                        "action": "x.parquet",
                        "prompt": "Strategy(Control: test.)",
                    }
                ],
            )
            metadata = validate_metadata(VARIANTS["v2"], metadata_path)
            shared = root / "shared"
            (shared / "video").mkdir(parents=True)
            (shared / "first_frame").mkdir()
            cache = root / "v2"
            cache.mkdir()
            (cache / "video").symlink_to(shared / "video")
            (cache / "first_frame").symlink_to(shared / "first_frame")
            (cache / "t5").mkdir()
            manifest = {
                "csv_md5": md5_file(metadata_path),
                "num_rows": 1,
                "config": {
                    "height": 480,
                    "width": 608,
                    "num_frames": 101,
                    "height_division_factor": 16,
                    "width_division_factor": 16,
                    "use_csv_prompt": True,
                    "prompt_column": "prompt",
                },
                "failed_rows": [],
                "rows": [{}],
                "rebuttal": {"metadata_sha256": metadata.sha256},
            }
            (cache / "manifest.json").write_text(json.dumps(manifest))
            audit = validate_cache(
                cache,
                metadata,
                height=480,
                width=608,
                num_frames=101,
            )
            self.assertEqual(Path(audit.video_dir), (shared / "video").resolve())
            self.assertEqual(
                Path(audit.first_frame_dir), (shared / "first_frame").resolve()
            )


class FormalConfigurationTests(unittest.TestCase):
    def test_approved_defaults_pass_and_drift_fails(self):
        self.assertEqual(FORMAL_DEFAULTS["num_processes"], 6)
        self.assertEqual(
            FORMAL_NUM_PROCESSES,
            {"v1": 6, "v2": 4, "v3": 6, "isaac_v1": 4},
        )
        values = dict(FORMAL_DEFAULTS)
        values["expected_num_processes"] = values.pop("num_processes")
        values["variant"] = "v1"
        args = SimpleNamespace(**values)
        validate_formal_hyperparameters(args)
        args.learning_rate = 1e-4
        with self.assertRaisesRegex(ValueError, "approved recipe"):
            validate_formal_hyperparameters(args)


if __name__ == "__main__":
    unittest.main()
