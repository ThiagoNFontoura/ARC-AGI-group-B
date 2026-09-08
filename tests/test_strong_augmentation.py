import unittest

from models.data_augmentation_baseline.strong_augmentation import (
    D4_GEOMETRIES,
    make_color_permutations,
    make_views,
    select_inference_orders,
    select_training_variants,
)


class StrongAugmentationTest(unittest.TestCase):
    def test_d4_has_eight_distinct_reversible_views_for_rectangular_grid(self) -> None:
        grid = [[0, 1, 2], [3, 4, 5]]
        transformed = []
        for geometry in D4_GEOMETRIES:
            view = geometry.apply(grid)
            transformed.append(tuple(tuple(row) for row in view))
            self.assertEqual(geometry.inverse(view), grid)
        self.assertEqual(len(set(transformed)), 8)

    def test_color_permutations_are_seeded_distinct_and_reversible(self) -> None:
        first = make_color_permutations(4, seed=17)
        second = make_color_permutations(4, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len({spec.mapping for spec in first}), 4)
        grid = [[0, 1, 9], [4, 2, 0]]
        for spec in first:
            self.assertEqual(spec.mapping[0], 0)
            self.assertEqual(spec.inverse(spec.apply(grid)), grid)

    def test_view_product_combines_d4_and_colors(self) -> None:
        views = make_views(color_permutations=4, seed=9)
        self.assertEqual(len(views), 32)
        grid = [[0, 1, 2], [3, 4, 5]]
        for view in views:
            self.assertEqual(view.inverse(view.apply(grid)), grid)

    def test_training_plan_is_capped_anchored_and_deterministic(self) -> None:
        plan, stats = select_training_variants(
            4,
            4,
            max_examples=256,
            color_permutations=4,
            identity_fraction=0.25,
            seed=42,
        )
        repeated, repeated_stats = select_training_variants(
            4,
            4,
            max_examples=256,
            color_permutations=4,
            identity_fraction=0.25,
            seed=42,
        )
        self.assertEqual(plan, repeated)
        self.assertEqual(stats, repeated_stats)
        self.assertEqual(stats.canonical_orderings, 48)
        self.assertEqual(stats.candidate_variants, 1536)
        self.assertEqual(stats.selected_variants, 256)
        self.assertEqual(stats.original_anchors, 48)

    def test_two_pair_tasks_receive_strong_ttt_items(self) -> None:
        plan, stats = select_training_variants(
            2,
            4,
            max_examples=256,
            color_permutations=4,
            identity_fraction=0.25,
            seed=42,
            minimum_group_size=2,
        )
        self.assertEqual(stats.canonical_orderings, 2)
        self.assertEqual(len(plan), 64)

    def test_inference_orders_include_original_and_are_distinct(self) -> None:
        orders = select_inference_orders(4, 3, seed=42)
        self.assertEqual(orders[0], (0, 1, 2, 3))
        self.assertEqual(len(orders), 3)
        self.assertEqual(len(set(orders)), 3)


if __name__ == "__main__":
    unittest.main()
