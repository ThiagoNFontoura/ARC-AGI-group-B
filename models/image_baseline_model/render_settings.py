"""Centralized render settings.

Edit this file to quickly test different visual styles for ARC images.
"""

from __future__ import annotations

# Colors are RGB tuples.
BACKGROUND_COLOR = (255, 255, 255)
GRID_COLOR = (0, 0, 0)
ZERO_COLOR = BACKGROUND_COLOR

# Known ARC values. Values not listed here are auto-generated with stable colors.
COLOR_BY_VALUE = {
    1: (31, 119, 180),
    2: (255, 127, 14),
    3: (44, 160, 44),
    4: (214, 39, 40),
    5: (148, 103, 189),
    6: (140, 86, 75),
    7: (227, 119, 194),
    8: (127, 127, 127),
    9: (188, 189, 34),
}

CELL_SIZE = 1
GRID_LINE_WIDTH = 0
IMAGE_MARGIN = 0
