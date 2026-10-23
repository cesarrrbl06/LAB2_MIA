"""Marker-controlled watershed on a gradient-magnitude relief.

SKELETON: implement each function (TODO).
"""
from typing import Sequence, Tuple

import numpy as np

BACKGROUND, FOREGROUND = 1, 2  # marker labels


def gradient_map(img: np.ndarray, method: str = "sobel", radius: int = 1) -> np.ndarray:
    """Gradient magnitude used as relief: 'sobel' or 'morphological' (dilation - erosion).

    Hints: skimage.filters.sobel, skimage.morphology.dilation / erosion / disk.
    """
    raise NotImplementedError


def build_markers(shape: Tuple[int, int], fg_points: Sequence[Tuple[int, int]],
                  bg_points: Sequence[Tuple[int, int]], fg_radius: int = 4, bg_radius: int = 4) -> np.ndarray:
    """Marker image: 0 = unknown, 1 = background, 2 = foreground. Hint: skimage.draw.disk."""
    raise NotImplementedError


def run_watershed(img: np.ndarray, markers: np.ndarray, method: str = "sobel"):
    """Flood the gradient from the markers. Return (labels, foreground_mask, boundaries).

    Hints: skimage.segmentation.watershed, skimage.segmentation.find_boundaries.
    """
    raise NotImplementedError


def watershed_oversegmentation(img: np.ndarray, method: str = "sobel"):
    """Watershed WITHOUT markers (to show over-segmentation). Return (labels, n_regions)."""
    raise NotImplementedError
