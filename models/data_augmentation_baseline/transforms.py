"""Reusable reversible transformations for ARC grids and tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

Grid = list[list[int]]
Transform = Callable[[Grid], Grid]


def _copy_grid(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def identity(grid: Grid) -> Grid:
    return _copy_grid(grid)


def flip_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def flip_vertical(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in reversed(grid)]


def transpose(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(column) for column in zip(*grid)]


@dataclass(frozen=True)
class TransformSpec:
    """A named transformation and its inverse."""

    name: str
    apply: Transform
    inverse: Transform


TRANSFORM_REGISTRY: dict[str, TransformSpec] = {
    "identity": TransformSpec("identity", identity, identity),
    "flip_horizontal": TransformSpec("flip_horizontal", flip_horizontal, flip_horizontal),
    "flip_vertical": TransformSpec("flip_vertical", flip_vertical, flip_vertical),
    "transpose": TransformSpec("transpose", transpose, transpose),
}

DEFAULT_TRANSFORM_NAMES = tuple(TRANSFORM_REGISTRY)


def get_transformations(names: list[str] | None = None) -> list[TransformSpec]:
    """Return configured transforms while preserving deterministic order."""
    selected_names = names or list(DEFAULT_TRANSFORM_NAMES)
    unknown = [name for name in selected_names if name not in TRANSFORM_REGISTRY]
    if unknown:
        available = ", ".join(TRANSFORM_REGISTRY)
        raise ValueError(f"Unknown transformations: {', '.join(unknown)}. Available: {available}")
    return [TRANSFORM_REGISTRY[name] for name in selected_names]


def transform_task(task: dict[str, Any], transform: Transform) -> dict[str, Any]:
    """Apply a grid transform consistently to every input and known output."""
    transformed = dict(task)
    transformed["train"] = [
        _transform_case(case, transform)
        for case in task.get("train", [])
    ]
    transformed["test"] = [
        _transform_case(case, transform)
        for case in task.get("test", [])
    ]
    return transformed


def transform_outputs(outputs: Any, inverse: Transform) -> Any:
    """Map a model response back to the original orientation."""
    if not isinstance(outputs, list):
        return outputs
    return [inverse(output) if _is_grid(output) else output for output in outputs]


def _transform_case(case: Any, transform: Transform) -> Any:
    if not isinstance(case, dict):
        return case
    transformed_case = dict(case)
    if _is_grid(case.get("input")):
        transformed_case["input"] = transform(case["input"])
    if _is_grid(case.get("output")):
        transformed_case["output"] = transform(case["output"])
    return transformed_case


def _is_grid(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, list) and all(isinstance(cell, int) for cell in row) for row in value
    )
