"""Pre-processing (filtering, histogram modifications) and post-processing (morphology).

SKELETON: implement each function (TODO).
"""
from typing import Optional

import numpy as np


# ----------------------------- pre-processing: filtering --------------------------- #
def smooth_gaussian(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smoothing. Hint: skimage.filters.gaussian."""
    raise NotImplementedError


def smooth_median(img: np.ndarray, radius: int = 2) -> np.ndarray:
    """Median filter with a disk footprint. Hint: scipy.ndimage.median_filter, skimage.morphology.disk."""
    raise NotImplementedError


def smooth_bilateral(img: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 3.0) -> np.ndarray:
    """Edge-preserving smoothing. Hint: skimage.restoration.denoise_bilateral."""
    raise NotImplementedError


# ----------------------------- pre-processing: histogram --------------------------- #
def equalize_hist(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation. Hint: skimage.exposure.equalize_hist."""
    raise NotImplementedError


def clahe(img: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Adaptive histogram equalisation. Hint: skimage.exposure.equalize_adapthist."""
    raise NotImplementedError


def contrast_stretch(img: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Linear stretch between two percentiles. Hint: skimage.exposure.rescale_intensity."""
    raise NotImplementedError


# ----------------------------- post-processing: morphology ------------------------- #
def clean_mask(mask: np.ndarray, opening_radius: int = 0, closing_radius: int = 0,
               min_size: int = 0, hole_size: Optional[int] = None,
               largest_only: bool = False) -> np.ndarray:
    """Morphological clean-up of a binary mask (opening, closing, small objects, holes, largest component).

    Hints: scipy.ndimage.binary_opening / binary_closing / binary_fill_holes / label.
    """
    raise NotImplementedError
