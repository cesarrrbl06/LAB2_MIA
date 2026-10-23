"""Intensity-based segmentation: global (Otsu), multi-Otsu and manual thresholding.

SKELETON: implement each function (TODO).
"""
from typing import Dict, Optional

import numpy as np


def otsu(img: np.ndarray) -> float:
    """Global Otsu threshold. Hint: skimage.filters.threshold_otsu."""
    raise NotImplementedError


def multi_otsu(img: np.ndarray, classes: int = 3) -> np.ndarray:
    """Multi-level Otsu thresholds. Hint: skimage.filters.threshold_multiotsu."""
    raise NotImplementedError


def apply_threshold(img: np.ndarray, low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
    """Binary mask of the pixels with low <= I <= high (either bound may be None)."""
    raise NotImplementedError


def plot_histogram(img: np.ndarray, thresholds: Optional[Dict[str, float]] = None, bins: int = 128):
    """Histogram of the ROI with one vertical line per threshold (returns the figure)."""
    raise NotImplementedError
