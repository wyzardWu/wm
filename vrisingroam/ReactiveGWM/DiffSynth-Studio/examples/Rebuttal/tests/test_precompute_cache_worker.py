import unittest

from examples.Rebuttal.precompute_cache_worker import build_parser


class PrecomputeCacheWorkerCliTests(unittest.TestCase):
    def _required_arguments(self) -> list[str]:
        return [
            "--metadata",
            "metadata.csv",
            "--dataset_base",
            "dataset",
            "--cache_root",
            "cache",
            "--model_paths",
            '["t5.pth"]',
            "--tokenizer_path",
            "tokenizer",
            "--rank",
            "0",
            "--world_size",
            "8",
        ]

    def test_t5_only_is_explicit_opt_in(self):
        parser = build_parser()
        self.assertFalse(parser.parse_args(self._required_arguments()).t5_only)
        self.assertTrue(
            parser.parse_args([*self._required_arguments(), "--t5_only"]).t5_only
        )


if __name__ == "__main__":
    unittest.main()
