"""Strong, reversible ARC augmentation without changing the legacy strategy.

The legacy augmentation module intentionally remains small and stable. This
module provides the richer policy used by the mini-ARC TTT comparison:

* all eight D4 grid symmetries;
* deterministic colour permutations;
* deterministic demonstration-order variants;
* a bounded, identity-anchored sampler for per-task TTT examples.

Every spatial and colour operation is applied uniformly to every grid in one
task view. Predictions must be mapped back with StrongView.inverse.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable, Sequence

Grid = list[list[int]]
GridTransform = Callable[[Grid], Grid]


def _copy_grid(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def identity(grid: Grid) -> Grid:
    return _copy_grid(grid)


def rotate_90(grid: Grid) -> Grid:
    """Rotate a rectangular grid 90 degrees clockwise."""
    return [list(row) for row in zip(*reversed(grid))]


def rotate_180(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in reversed(grid)]


def rotate_270(grid: Grid) -> Grid:
    """Rotate a rectangular grid 270 degrees clockwise."""
    return [list(row) for row in reversed(list(zip(*grid)))]


def flip_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def flip_vertical(grid: Grid) -> Grid:
    return [list(row) for row in reversed(grid)]


def transpose(grid: Grid) -> Grid:
    return [list(column) for column in zip(*grid)]


def anti_transpose(grid: Grid) -> Grid:
    """Reflect a grid across its anti-diagonal."""
    return rotate_180(transpose(grid))


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    apply: GridTransform
    inverse: GridTransform


D4_GEOMETRIES: tuple[GeometrySpec, ...] = (
    GeometrySpec("identity", identity, identity),
    GeometrySpec("rotate_90", rotate_90, rotate_270),
    GeometrySpec("rotate_180", rotate_180, rotate_180),
    GeometrySpec("rotate_270", rotate_270, rotate_90),
    GeometrySpec("flip_horizontal", flip_horizontal, flip_horizontal),
    GeometrySpec("flip_vertical", flip_vertical, flip_vertical),
    GeometrySpec("transpose", transpose, transpose),
    GeometrySpec("anti_transpose", anti_transpose, anti_transpose),
)


@dataclass(frozen=True)
class ColorPermutation:
    name: str
    mapping: tuple[int, ...]
    inverse_mapping: tuple[int, ...]

    def apply(self, grid: Grid) -> Grid:
        return [[self.mapping[value] for value in row] for row in grid]

    def inverse(self, grid: Grid) -> Grid:
        return [[self.inverse_mapping[value] for value in row] for row in grid]


@dataclass(frozen=True)
class StrongView:
    geometry: GeometrySpec
    colors: ColorPermutation

    @property
    def name(self) -> str:
        return f"{self.geometry.name}+{self.colors.name}"

    @property
    def is_original(self) -> bool:
        return self.geometry.name == "identity" and self.colors.name == "identity"

    def apply(self, grid: Grid) -> Grid:
        return self.colors.apply(self.geometry.apply(grid))

    def inverse(self, grid: Grid) -> Grid:
        return self.geometry.inverse(self.colors.inverse(grid))


@dataclass(frozen=True)
class TrainingVariant:
    ordering: tuple[int, ...]
    view: StrongView


@dataclass(frozen=True)
class TrainingVariantStats:
    canonical_orderings: int
    candidate_variants: int
    selected_variants: int
    original_anchors: int


def _color_permutation(mapping: Sequence[int], name: str) -> ColorPermutation:
    mapping_tuple = tuple(mapping)
    if sorted(mapping_tuple) != list(range(10)):
        raise ValueError("an ARC colour mapping must be a permutation of 0..9")
    inverse = [0] * 10
    for source, destination in enumerate(mapping_tuple):
        inverse[destination] = source
    return ColorPermutation(name, mapping_tuple, tuple(inverse))


def make_color_permutations(
    count: int,
    *,
    seed: int,
    preserve_zero: bool = True,
) -> list[ColorPermutation]:
    """Return identity plus deterministic, distinct random colour mappings."""
    if count < 1:
        raise ValueError("at least one colour permutation is required")
    results = [_color_permutation(range(10), "identity")]
    if count == 1:
        return results

    rng = random.Random(seed)
    seen = {results[0].mapping}
    attempts = 0
    while len(results) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not generate enough distinct colour permutations")
        if preserve_zero:
            foreground = list(range(1, 10))
            rng.shuffle(foreground)
            mapping = (0, *foreground)
        else:
            shuffled = list(range(10))
            rng.shuffle(shuffled)
            mapping = tuple(shuffled)
        if mapping in seen:
            continue
        seen.add(mapping)
        results.append(_color_permutation(mapping, f"color_{len(results)}"))
    return results


def make_views(
    *,
    color_permutations: int,
    seed: int,
    preserve_zero: bool = True,
) -> list[StrongView]:
    colors = make_color_permutations(
        color_permutations,
        seed=seed,
        preserve_zero=preserve_zero,
    )
    return [StrongView(geometry, color) for geometry in D4_GEOMETRIES for color in colors]


def _canonical_orderings(
    pool_size: int,
    maximum_group_size: int,
    minimum_group_size: int,
) -> list[tuple[int, ...]]:
    upper = min(pool_size, maximum_group_size)
    if upper < minimum_group_size:
        return []
    indices = range(pool_size)
    return [
        ordering
        for length in range(minimum_group_size, upper + 1)
        for ordering in itertools.permutations(indices, length)
    ]


def select_training_variants(
    pool_size: int,
    maximum_group_size: int,
    *,
    max_examples: int,
    color_permutations: int,
    identity_fraction: float,
    seed: int,
    preserve_zero: bool = True,
    minimum_group_size: int = 3,
) -> tuple[list[TrainingVariant], TrainingVariantStats]:
    """Build a deterministic, bounded strong-augmentation training plan.

    Original identity/no-colour items are selected first as anchors. The
    remainder is sampled from the full D4 x colour candidate set.
    """
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    if not 0.0 <= identity_fraction <= 1.0:
        raise ValueError("identity_fraction must be in [0, 1]")

    orderings = _canonical_orderings(
        pool_size,
        maximum_group_size,
        minimum_group_size,
    )
    views = make_views(
        color_permutations=color_permutations,
        seed=seed,
        preserve_zero=preserve_zero,
    )
    candidates = [TrainingVariant(ordering, view) for view in views for ordering in orderings]
    if len(candidates) <= max_examples:
        return candidates, TrainingVariantStats(
            canonical_orderings=len(orderings),
            candidate_variants=len(candidates),
            selected_variants=len(candidates),
            original_anchors=sum(variant.view.is_original for variant in candidates),
        )

    rng = random.Random(seed)
    anchors = [variant for variant in candidates if variant.view.is_original]
    augmented = [variant for variant in candidates if not variant.view.is_original]
    anchor_count = min(len(anchors), round(max_examples * identity_fraction))
    selected_anchors = rng.sample(anchors, anchor_count)
    remaining = max_examples - anchor_count
    selected = [*selected_anchors, *rng.sample(augmented, remaining)]
    rng.shuffle(selected)
    return selected, TrainingVariantStats(
        canonical_orderings=len(orderings),
        candidate_variants=len(candidates),
        selected_variants=len(selected),
        original_anchors=anchor_count,
    )


def select_inference_orders(
    demonstration_count: int,
    count: int,
    *,
    seed: int,
) -> list[tuple[int, ...]]:
    """Select distinct deterministic orders, always starting with the original."""
    if count < 1:
        raise ValueError("at least one inference order is required")
    original = tuple(range(demonstration_count))
    all_orders = list(itertools.permutations(range(demonstration_count)))
    alternatives = [ordering for ordering in all_orders if ordering != original]
    random.Random(seed).shuffle(alternatives)
    return [original, *alternatives[: max(0, count - 1)]]
