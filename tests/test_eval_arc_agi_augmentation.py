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
    from arc_prize.eval_arc_agi import (
        _hierarchical_vote_predictions,
        _load_extra_examples,
        _strong_ttt_examples,
        _ttt_examples,
    )
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

    def test_strong_ttt_supports_two_pairs_and_caps_four_pairs(self) -> None:
        two_pair_examples, two_pair_stats = _strong_ttt_examples(
            pairs(2),
            self.config,
            max_examples=256,
            color_permutations=4,
            identity_fraction=0.25,
            preserve_zero=True,
            seed=42,
        )
        four_pair_examples, four_pair_stats = _strong_ttt_examples(
            pairs(4),
            self.config,
            max_examples=256,
            color_permutations=4,
            identity_fraction=0.25,
            preserve_zero=True,
            seed=42,
        )
        self.assertEqual(len(two_pair_examples), 64)
        self.assertEqual(two_pair_stats.canonical_orderings, 2)
        self.assertEqual(len(four_pair_examples), 256)
        self.assertEqual(four_pair_stats.candidate_variants, 1536)

    def test_hierarchical_vote_prefers_cross_geometry_consistency(self) -> None:
        repeated = torch.ones((12, 12), dtype=torch.int)
        candidates = []
        for geometry in ("identity", "rotate_90", "rotate_180"):
            candidates.extend(
                {
                    "geometry": geometry,
                    "color": color,
                    "order_index": order,
                    "prediction": repeated,
                }
                for color, order in (("identity", 0), ("color_1", 1))
            )
        winner, details = _hierarchical_vote_predictions(candidates)
        self.assertTrue(torch.equal(winner, repeated))
        self.assertEqual(details["winner_votes"], 3)


if __name__ == "__main__":
    unittest.main()
