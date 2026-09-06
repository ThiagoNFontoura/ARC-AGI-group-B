import json
from typing import Any


def _grid_properties(grid: Any) -> dict[str, Any]:
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) for row in grid):
        return {"valid_grid": False}
    width = len(grid[0])
    if width == 0 or any(len(row) != width for row in grid):
        return {"valid_grid": False}
    if not all(isinstance(cell, int) for row in grid for cell in row):
        return {"valid_grid": False}

    height = len(grid)
    colors = sorted({cell for row in grid for cell in row})
    counts = {str(color): sum(row.count(color) for row in grid) for color in colors}
    background = max(colors, key=lambda color: (counts[str(color)], -color))

    return {
        "valid_grid": True,
        "height": height,
        "width": width,
        "colors": colors,
        "background_color": background,
    }


def _constant_properties(properties: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in properties if item.get("valid_grid")]
    if not valid:
        return {}
    constants = {}
    for name in valid[0]:
        values = [item.get(name) for item in valid]
        if all(value == values[0] for value in values):
            constants[name] = values[0]
    return constants


def analyze_task_invariants(train: list[Any]) -> dict[str, Any]:
    inputs = [_grid_properties(case.get("input")) for case in train if isinstance(case, dict)]
    outputs = [_grid_properties(case.get("output")) for case in train if isinstance(case, dict)]
    return {
        "inputs": _constant_properties(inputs),
        "outputs": _constant_properties(outputs),
    }


def validate_grid_invariants(grid: Any, constants: dict[str, Any]) -> bool:
    properties = _grid_properties(grid)
    return properties.get("valid_grid") is True and all(
        properties.get(name) == value for name, value in constants.items()
    )


def format_invariants_for_prompt(analysis: dict[str, Any]) -> str:
    return json.dumps(analysis, ensure_ascii=True, separators=(",", ":"))