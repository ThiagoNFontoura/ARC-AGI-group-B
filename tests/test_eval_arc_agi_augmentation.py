import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

if torch is not None:
    from arc_prize.data import ARCDatasetParams
    from arc_prize.eval_arc_agi import _load_extra_examples, _ttt_examples
    from models.data_augmentation_baseline.transforms import get_transformations


def pairs(count: int) -> list[dict]:
    return [
        {"input": [[index % 10]], "output": [[(index + 1) % 10]]}
        for index in range(count)
    ]


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class EvalArcAugmentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ARCDatasetParams(max_grid_size=12, max_train_grids=4, color_offset=1)

    def test_geometric_views_multiply_original_ttt_items(self) -> None:
        original = _ttt_examples(pairs(4), self.config)
        augmented = _ttt_examples(
            pairs(4),
            self.config,
            get_transformations(
                ["identity", "flip_horizontal", "flip_vertical", "transpose"]
            ),
        )
        self.assertEqual(len(original), 48)
        self.assertEqual(len(augmented), 192)

    def test_larger_pool_caps_group_size_but_uses_all_combinations(self) -> None:
        # P(5, 3) + P(5, 4) = 60 + 120.
        self.assertEqual(len(_ttt_examples(pairs(5), self.config)), 180)
        capped = _ttt_examples(pairs(5), self.config, max_examples=25, seed=7)
        repeated = _ttt_examples(pairs(5), self.config, max_examples=25, seed=7)
        self.assertEqual(len(capped), 25)
        self.assertTrue(
            all(torch.equal(left[0], right[0]) for left, right in zip(capped, repeated))
        )

    def test_extra_loader_skips_fallback_and_oversized_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "generated.json").write_text(
                json.dumps([*pairs(2), {"input": [[0] * 13], "output": [[1]]}])
            )
            loaded, stats = _load_extra_examples(root, "generated", 12)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(stats["rejected"], 1)

            (root / "fallback.json").write_text(
                json.dumps({"train": pairs(3), "test": [{"input": [[0]]}]})
            )
            loaded, stats = _load_extra_examples(root, "fallback", 12)
            self.assertEqual(loaded, [])
            self.assertEqual(stats["status"], "original_task_fallback")


if __name__ == "__main__":
    unittest.main()
