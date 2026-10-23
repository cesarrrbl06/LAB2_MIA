"""Marker-controlled watershed on a gradient-magnitude relief."""
from typing import Sequence, Tuple

import numpy as np
from skimage.draw import disk as draw_disk
from skimage.filters import sobel
from skimage.morphology import dilation, disk, erosion
from skimage.segmentation import find_boundaries, watershed

BACKGROUND, FOREGROUND = 1, 2  # marker labels


def gradient_map(img: np.ndarray, method: str = "sobel", radius: int = 1) -> np.ndarray:
    """Gradient magnitude used as relief: 'sobel' or 'morphological' (dilation - erosion)."""
    img = np.asarray(img, dtype=float)
    if method == "sobel":
        return sobel(img)
    if method == "morphological":
        fp = disk(radius)
        return dilation(img, fp) - erosion(img, fp)
    raise ValueError("method must be 'sobel' or 'morphological'")


def build_markers(shape: Tuple[int, int], fg_points: Sequence[Tuple[int, int]],
                  bg_points: Sequence[Tuple[int, int]], fg_radius: int = 4, bg_radius: int = 4) -> np.ndarray:
    """Marker image: 0 = unknown, 1 = background, 2 = foreground."""
    markers = np.zeros(shape, dtype=np.int32)
    for (r, c) in bg_points:
        rr, cc = draw_disk((r, c), bg_radius, shape=shape)
        markers[rr, cc] = BACKGROUND
    for (r, c) in fg_points:
        rr, cc = draw_disk((r, c), fg_radius, shape=shape)
        markers[rr, cc] = FOREGROUND
    return markers


def run_watershed(img: np.ndarray, markers: np.ndarray, method: str = "sobel"):
    """Flood the gradient from the markers. Return (labels, foreground_mask, boundaries)."""
    relief = gradient_map(img, method=method)
    labels = watershed(relief, markers=markers)
    foreground = labels == FOREGROUND
    boundaries = find_boundaries(labels, mode="inner")
    return labels, foreground, boundaries


def watershed_oversegmentation(img: np.ndarray, method: str = "sobel"):
    """Watershed WITHOUT markers (to show over-segmentation). Return (labels, n_regions)."""
    relief = gradient_map(img, method=method)
    labels = watershed(relief)  # every local minimum becomes a marker
    return labels, int(len(np.unique(labels)))
