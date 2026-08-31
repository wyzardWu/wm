import os
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from examples.Rebuttal.supplemental_claim_guard import install_boundary_guards
from examples.Rebuttal.supplemental_vae_worker import (
    claim_path,
    release_claim,
    try_acquire_claim,
)


class SupplementalClaimTests(unittest.TestCase):
    def test_claim_is_exclusive_and_releasable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = claim_path(directory, "abcdef")
            payload = {"worker_id": "test"}
            self.assertTrue(
                try_acquire_claim(
                    path,
                    payload,
                    stale_after_seconds=60,
                )
            )
            self.assertFalse(
                try_acquire_claim(
                    path,
                    payload,
                    stale_after_seconds=60,
                )
            )
            release_claim(path)
            self.assertTrue(
                try_acquire_claim(
                    path,
                    payload,
                    stale_after_seconds=60,
                )
            )

    def test_stale_claim_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = claim_path(directory, "123456")
            path.parent.mkdir(parents=True)
            path.write_text("stale", encoding="utf-8")
            old = time.time() - 120
            os.utime(path, (old, old))
            self.assertTrue(
                try_acquire_claim(
                    path,
                    {"worker_id": "replacement"},
                    stale_after_seconds=60,
                )
            )
            release_claim(path)

    def test_boundary_guard_reserves_only_lower_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            pd.DataFrame(
                [
                    {"video": "a.mp4"},
                    {"video": "b.mp4"},
                    {"video": "c.mp4"},
                ]
            ).to_csv(metadata, index=False)
            report = install_boundary_guards(
                metadata=metadata,
                cache_root=root / "cache",
                below_csv_index=2,
            )
            self.assertEqual(report["created"], 2)
            self.assertEqual(report["already_complete"], 0)
            claims = list((root / "cache/.supplemental_claims").rglob("*.claim"))
            self.assertEqual(len(claims), 2)


if __name__ == "__main__":
    unittest.main()
