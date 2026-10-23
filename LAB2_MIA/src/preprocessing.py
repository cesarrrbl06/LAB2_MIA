"""Pre-processing (filtering, histogram modifications) and post-processing (morphology)."""
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral
from skimage.util import img_as_float


# ----------------------------- pre-processing: filtering --------------------------- #
def smooth_gaussian(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian smoothing. Hint: skimage.filters.gaussian."""
    return filters.gaussian(img_as_float(img), sigma=sigma, preserve_range=True)


def smooth_median(img: np.ndarray, radius: int = 2) -> np.ndarray:
    """Median filter with a disk footprint. Hint: scipy.ndimage.median_filter, skimage.morphology.disk."""
    return ndi.median_filter(img_as_float(img), footprint=disk(radius))


def smooth_bilateral(img: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 3.0) -> np.ndarray:
    """Edge-preserving smoothing. Hint: skimage.restoration.denoise_bilateral."""
    return denoise_bilateral(img_as_float(img), sigma_color=sigma_color, sigma_spatial=sigma_spatial)


# ----------------------------- pre-processing: histogram --------------------------- #
def equalize_hist(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation. Hint: skimage.exposure.equalize_hist."""
    return exposure.equalize_hist(img_as_float(img))


def clahe(img: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Adaptive histogram equalisation. Hint: skimage.exposure.equalize_adapthist."""
    return exposure.equalize_adapthist(np.clip(img_as_float(img), 0.0, 1.0), clip_limit=clip_limit)


def contrast_stretch(img: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Linear stretch between two percentiles. Hint: skimage.exposure.rescale_intensity."""
    img = img_as_float(img)
    lo, hi = np.percentile(img, (p_low, p_high))
    return exposure.rescale_intensity(img, in_range=(lo, hi), out_range=(0.0, 1.0))


# ----------------------------- post-processing: morphology ------------------------- #
def clean_mask(mask: np.ndarray, opening_radius: int = 0, closing_radius: int = 0,
               min_size: int = 0, hole_size: Optional[int] = None,
               largest_only: bool = False) -> np.ndarray:
    """Morphological clean-up of a binary mask (opening, closing, small objects, holes, largest component).

    hole_size: None -> holes are not filled; 0 -> fill every hole; > 0 -> fill holes smaller than it.
    """
    out = np.asarray(mask, dtype=bool).copy()
    if opening_radius > 0:
        out = ndi.binary_opening(out, structure=disk(opening_radius))
    if closing_radius > 0:
        # pad so that closing does not erode objects touching the border
        pad = closing_radius
        out = ndi.binary_closing(np.pad(out, pad), structure=disk(closing_radius))[pad:-pad, pad:-pad]
    if min_size > 0 and out.any():
        labels, n = ndi.label(out)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_size
        keep[0] = False
        out = keep[labels]
    if hole_size is not None:
        filled = ndi.binary_fill_holes(out)
        if hole_size > 0:
            holes = filled & ~out
            labels, n = ndi.label(holes)
            sizes = np.bincount(labels.ravel())
            small = sizes < hole_size
            small[0] = False
            out = out | small[labels]
        else:
            out = filled
    if largest_only and out.any():
        labels, n = ndi.label(out)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        out = labels == int(np.argmax(sizes))
    return out
