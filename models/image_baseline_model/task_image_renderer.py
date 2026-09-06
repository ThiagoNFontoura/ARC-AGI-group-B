from __future__ import annotations

import colorsys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from . import render_settings

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class RenderStyle:
    background_color: RGB
    grid_color: RGB
    zero_color: RGB
    color_by_value: dict[int, RGB]
    cell_size: int
    grid_line_width: int
    image_margin: int


def default_render_style() -> RenderStyle:
    return RenderStyle(
        background_color=render_settings.BACKGROUND_COLOR,
        grid_color=render_settings.GRID_COLOR,
        zero_color=render_settings.ZERO_COLOR,
        color_by_value=dict(render_settings.COLOR_BY_VALUE),
        cell_size=render_settings.CELL_SIZE,
        grid_line_width=render_settings.GRID_LINE_WIDTH,
        image_margin=render_settings.IMAGE_MARGIN,
    )


def _sanitize_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("_")
    return clean or "task"


def _is_grid(grid: Any) -> bool:
    if not isinstance(grid, list) or not grid:
        return False
    if not all(isinstance(row, list) and row for row in grid):
        return False
    cols = len(grid[0])
    if not all(len(row) == cols for row in grid):
        return False
    for row in grid:
        for value in row:
            if not isinstance(value, int):
                return False
    return True


def _auto_color(value: int) -> RGB:
    # Stable color per value while avoiding near-black and near-white tones.
    hue = ((value * 53) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


def _value_color(value: int, style: RenderStyle) -> RGB:
    if value == 0:
        return style.zero_color
    if value in style.color_by_value:
        return style.color_by_value[value]
    return _auto_color(value)


def render_grid_image(grid: list[list[int]], style: RenderStyle) -> Image.Image:
    rows = len(grid)
    cols = len(grid[0])
    cell = style.cell_size
    line = style.grid_line_width

    width = cols * cell + (cols + 1) * line
    height = rows * cell + (rows + 1) * line

    image = Image.new("RGB", (width, height), color=style.grid_color)
    draw = ImageDraw.Draw(image)

    for r in range(rows):
        for c in range(cols):
            x0 = line + c * (cell + line)
            y0 = line + r * (cell + line)
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            draw.rectangle((x0, y0, x1, y1), fill=_value_color(grid[r][c], style))

    return image


def _save_grid(
    grid: Any,
    output_path: Path,
    style: RenderStyle,
) -> bool:
    if not _is_grid(grid):
        return False
    img = render_grid_image(grid, style)
    img.save(output_path)
    return True


def render_single_task(task: dict[str, Any], output_root: Path, style: RenderStyle) -> dict[str, Any]:
    task_name = str(task.get("task_name") or "task")
    safe_name = _sanitize_name(task_name)
    task_dir = output_root / safe_name
    task_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped = 0

    for split in ("train", "test"):
        cases = task.get(split, [])
        if not isinstance(cases, list):
            continue

        for idx, case in enumerate(cases):
            if not isinstance(case, dict):
                skipped += 1
                continue

            input_path = task_dir / f"{split}_{idx:02d}_input.png"
            if _save_grid(case.get("input"), input_path, style):
                rendered += 1
            else:
                skipped += 1

            if "output" in case:
                output_path = task_dir / f"{split}_{idx:02d}_output.png"
                if _save_grid(case.get("output"), output_path, style):
                    rendered += 1
                else:
                    skipped += 1

    return {
        "task_name": task_name,
        "output_dir": str(task_dir),
        "rendered_images": rendered,
        "skipped_items": skipped,
    }


def render_tasks_folder_parallel(
    tasks: list[dict[str, Any]],
    output_root: Path,
    style: RenderStyle | None = None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    style = style or default_render_style()
    output_root.mkdir(parents=True, exist_ok=True)

    if not tasks:
        return []

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(render_single_task, task, output_root, style) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: str(item.get("task_name", "")))
    return results
