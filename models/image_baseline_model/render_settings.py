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
    0: (255,255,255),
    1: (30, 147, 255),
    2: (249, 60, 49),
    3: (78, 234, 58),
    4: (255,216,77),
    5: (153,153,153),
    6: (229,58,163),
    7: (255,138,31),
    8: (79,227,240),
    9: (146,18,49),
}

CELL_SIZE = 10
GRID_LINE_WIDTH = 1
IMAGE_MARGIN = 0
