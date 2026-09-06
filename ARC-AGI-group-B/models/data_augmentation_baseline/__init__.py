"""Baseline ARC solver with modular geometric test-time augmentation."""

from .strategy import AugmentedInference
from .transforms import TransformSpec, get_transformations

__all__ = ["AugmentedInference", "TransformSpec", "get_transformations"]
