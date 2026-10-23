"""Shared helpers: project paths, image loading, ROI handling, plotting and export.

SKELETON: only the paths are defined. Implement each function (TODO).
"""
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.util import img_as_float

# --------------------------------------------------------------------------- #
# Project paths (relative to the project root, independent of the working dir)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "raw_images"
DATA_GT = PROJECT_ROOT / "data" / "ground_truth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_TABLES = RESULTS_DIR / "tables"
RESULTS_SEG = RESULTS_DIR / "segmentations"

ROI = Tuple[int, int, int, int]  # (ymin, ymax, xmin, xmax) -> img[ymin:ymax, xmin:xmax]


def ensure_dirs() -> None:
    """Create the output folders if they do not exist (Path.mkdir)."""
    raise NotImplementedError


def set_seed(seed: int = 42) -> int:
    """Fix random / numpy seeds for reproducibility and return the seed."""
    raise NotImplementedError


def load_image(filename: str, folder: Path = DATA_RAW) -> np.ndarray:
    """Load a clinical image (.png or .nrrd) as a 2-D float array in [0, 1].

    Hints: skimage.io.imread, skimage.util.img_as_float, nrrd.read (pynrrd, [x, y] order).
    """
    raise NotImplementedError


def crop_roi(img: np.ndarray, roi: ROI) -> np.ndarray:
    """Return the sub-image defined by the bounding box ``roi``."""
    raise NotImplementedError


def roi_dimensions(roi: ROI) -> Tuple[int, int, int]:
    """Return (height, width, N_ROI) with N_ROI = number of pixels of the ROI."""
    raise NotImplementedError


def plot_roi_overview(img: np.ndarray, roi: ROI, title: str = ""):
    """Figure: full image with a red rectangle on the ROI vs. the cropped sub-image.

    Hint: matplotlib.patches.Rectangle.
    """
    raise NotImplementedError


def show_overlays(img: np.ndarray, masks: Dict[str, np.ndarray], gt: Optional[np.ndarray] = None):
    """Grid with the image and the contour of each mask (and of the ground truth, if given).

    Hint: ax.contour(mask, levels=[0.5]).
    """
    raise NotImplementedError


def save_figure(fig, filename: str, dpi: int = 300) -> Path:
    """Save a figure into results/figures at 300 DPI."""
    raise NotImplementedError


def save_table(df, filename: str) -> Path:
    """Save a DataFrame as CSV into results/tables."""
    raise NotImplementedError
