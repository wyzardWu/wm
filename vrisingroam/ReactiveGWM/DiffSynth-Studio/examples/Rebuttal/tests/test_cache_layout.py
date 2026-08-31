from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from examples.ReactiveGWM.model_training.cached_dataset import (
    _shard_path,
    t5_cache_key,
    video_cache_key,
)
from examples.Rebuttal.cache_layout import (
    finalize_manifest,
    initialize_layout,
    variant_cache_root,
)
from examples.Rebuttal.preflight import validate_cache, validate_metadata
from examples.Rebuttal.variants import VARIANTS


class CacheLayoutTests(unittest.TestCase):
    def test_finalize_shared_vae_and_variant_t5_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_base = root / "cache"
            layout = initialize_layout(cache_base)
            initialize_layout(cache_base)  # Idempotent.
            self.assertEqual(
                layout["variants"]["v1"]["video"],
                layout["variants"]["v2"]["video"],
            )
            self.assertNotEqual(
                layout["variants"]["v1"]["t5"],
                layout["variants"]["v2"]["t5"],
            )

            metadata_path = root / "metadata.csv"
            with metadata_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("video", "action", "prompt")
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video": "clip/video.mp4",
                        "action": "clip/actions.parquet",
                        "prompt": "Strategy(Control: test.)",
                    }
                )
            metadata = validate_metadata(VARIANTS["v2"], metadata_path)
            variant_root = variant_cache_root(cache_base, "v2")
            video_hash = video_cache_key("clip/video.mp4", 16, 16, 5, 16, 16, False)
            first_hash = video_cache_key("clip/video.mp4", 16, 16, 5, 16, 16, True)
            prompt_hash = t5_cache_key("Strategy(Control: test.)")
            empty_hash = t5_cache_key("")
            tensors = {
                ("video", video_hash): torch.zeros(1, 2, 2, 2, 2),
                ("first_frame", first_hash): torch.zeros(1, 2, 1, 2, 2),
                ("t5", prompt_hash): torch.zeros(1, 4, 8),
                ("t5", empty_hash): torch.zeros(1, 4, 8),
            }
            for (kind, digest), tensor in tensors.items():
                path = _shard_path(variant_root, kind, digest)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tensor, path)

            vae = root / "vae.pth"
            t5 = root / "t5.pth"
            vae.write_bytes(b"fake vae")
            t5.write_bytes(b"fake t5")
            manifest = finalize_manifest(
                cache_base=cache_base,
                variant="v2",
                metadata_path=metadata_path,
                dataset_base=root,
                vae_path=vae,
                t5_path=t5,
                height=16,
                width=16,
                num_frames=5,
            )
            self.assertEqual(manifest["num_rows"], 1)
            self.assertEqual(manifest["rebuttal"]["variant"], "v2")
            self.assertEqual(manifest["config"]["vae_upsampling_factor"], 8)
            cache_audit = validate_cache(
                variant_root,
                metadata,
                height=16,
                width=16,
                num_frames=5,
            )
            self.assertEqual(cache_audit.rows, 1)
            self.assertEqual(
                Path(cache_audit.video_dir),
                (cache_base / "shared_vae/video").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
