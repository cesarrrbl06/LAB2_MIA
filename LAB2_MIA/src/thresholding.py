"""Intensity-based segmentation: global (Otsu), multi-Otsu and manual thresholding."""
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu, threshold_otsu


def otsu(img: np.ndarray) -> float:
    """Global Otsu threshold. Hint: skimage.filters.threshold_otsu."""
    return float(threshold_otsu(np.asarray(img)))


def multi_otsu(img: np.ndarray, classes: int = 3) -> np.ndarray:
    """Multi-level Otsu thresholds. Hint: skimage.filters.threshold_multiotsu."""
    return threshold_multiotsu(np.asarray(img), classes=classes)


def apply_threshold(img: np.ndarray, low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
    """Binary mask of the pixels with low <= I <= high (either bound may be None)."""
    img = np.asarray(img)
    mask = np.ones(img.shape, dtype=bool)
    if low is not None:
        mask &= img >= low
    if high is not None:
        mask &= img <= high
    return mask


def plot_histogram(img: np.ndarray, thresholds: Optional[Dict[str, float]] = None, bins: int = 128):
    """Histogram of the ROI with one vertical line per threshold (returns the figure)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.asarray(img).ravel(), bins=bins, color="gray", alpha=0.8)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, t) in enumerate((thresholds or {}).items()):
        ax.axvline(float(t), color=colors[i % len(colors)], linestyle="--", linewidth=2,
                   label=f"{name} = {float(t):.3f}")
    if thresholds:
        ax.legend()
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Number of pixels")
    ax.set_title("ROI histogram")
    fig.tight_layout()
    return fig
