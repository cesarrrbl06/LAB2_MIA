"""Shared helpers: project paths, image loading, ROI handling, plotting and export."""
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pandas as pd
from skimage.color import rgb2gray, rgba2rgb
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
    for folder in (DATA_GT, RESULTS_FIGURES, RESULTS_TABLES, RESULTS_SEG):
        folder.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> int:
    """Fix random / numpy seeds for reproducibility and return the seed."""
    random.seed(seed)
    np.random.seed(seed)
    return seed


def _to_float01(img: np.ndarray) -> np.ndarray:
    """Convert any grayscale/RGB/RGBA array to a 2-D float64 image in [0, 1]."""
    img = np.squeeze(np.asarray(img))
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = rgba2rgb(img)
        if img.shape[-1] == 3:
            img = rgb2gray(img)
        elif img.shape[-1] == 2:  # gray + alpha
            img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {img.shape}")
    if img.dtype.kind in "ui" or img.dtype == bool:
        img = img_as_float(img)
    img = img.astype(np.float64)
    lo, hi = float(img.min()), float(img.max())
    if lo < 0.0 or hi > 1.0:  # e.g. raw .nrrd intensities
        img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return img


def load_image(filename: str, folder: Path = DATA_RAW) -> np.ndarray:
    """Load a clinical image (.png or .nrrd) as a 2-D float array in [0, 1]."""
    path = Path(filename)
    if not path.is_absolute() and not path.exists():
        path = Path(folder) / filename
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.name.endswith(".nrrd"):
        data, _ = nrrd.read(str(path))
        data = np.squeeze(data)
        if data.ndim == 2:
            data = data.T  # nrrd stores [x, y] -> numpy [row, col]
        return _to_float01(data.astype(np.float64))
    return _to_float01(imread(str(path)))


def crop_roi(img: np.ndarray, roi: ROI) -> np.ndarray:
    """Return the sub-image defined by the bounding box ``roi``."""
    ymin, ymax, xmin, xmax = roi
    return img[ymin:ymax, xmin:xmax].copy()


def roi_dimensions(roi: ROI) -> Tuple[int, int, int]:
    """Return (height, width, N_ROI) with N_ROI = number of pixels of the ROI."""
    ymin, ymax, xmin, xmax = roi
    height, width = ymax - ymin, xmax - xmin
    return height, width, height * width


def to_roi_coords(points: Sequence[Tuple[int, int]], roi: ROI) -> List[Tuple[int, int]]:
    """Convert (row, col) points from full-image coordinates to ROI coordinates (checks they fall inside)."""
    ymin, ymax, xmin, xmax = roi
    out = []
    for r, c in points:
        if not (ymin <= r < ymax and xmin <= c < xmax):
            raise ValueError(f"Point {(r, c)} is outside the ROI {roi}")
        out.append((int(r - ymin), int(c - xmin)))
    return out


def plot_roi_overview(img: np.ndarray, roi: ROI, title: str = ""):
    """Figure: full image with a red rectangle on the ROI vs. the cropped sub-image."""
    ymin, ymax, xmin, xmax = roi
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].add_patch(mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         fill=False, edgecolor="red", linewidth=2))
    axes[0].set_title(f"{title} — full image {img.shape[0]}×{img.shape[1]}".strip(" —"))
    h, w, n = roi_dimensions(roi)
    axes[1].imshow(crop_roi(img, roi), cmap="gray")
    axes[1].set_title(f"ROI [{ymin}:{ymax}, {xmin}:{xmax}] — {h}×{w} = {n} px")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def show_overlays(img: np.ndarray, masks: Dict[str, np.ndarray], gt: Optional[np.ndarray] = None):
    """Grid with the image and the contour of each mask (and of the ground truth, if given)."""
    n = len(masks) + 1
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image")
    if gt is not None:
        axes[0].contour(np.asarray(gt, dtype=float), levels=[0.5], colors="lime", linewidths=1.5)
        axes[0].set_title("Image + ground truth")
    for ax, (name, mask) in zip(axes[1:], masks.items()):
        ax.imshow(img, cmap="gray")
        m = np.asarray(mask, dtype=float)
        if m.any():
            ax.contour(m, levels=[0.5], colors="red", linewidths=1.5)
        if gt is not None and np.any(gt):
            ax.contour(np.asarray(gt, dtype=float), levels=[0.5], colors="lime", linewidths=1, linestyles="--")
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def save_figure(fig, filename: str, dpi: int = 300) -> Path:
    """Save a figure into results/figures at 300 DPI."""
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = RESULTS_FIGURES / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(df, filename: str) -> Path:
    """Save a DataFrame as CSV into results/tables."""
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    path = RESULTS_TABLES / filename
    pd.DataFrame(df).to_csv(path, index=True)
    return path
