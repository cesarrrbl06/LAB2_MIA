"""Segmentation metrics against a ground-truth mask.

SKELETON: implement each function (TODO).
"""
from typing import Dict

import numpy as np
import pandas as pd


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice Similarity Coefficient: 2|A∩B| / (|A| + |B|)."""
    raise NotImplementedError


def hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hausdorff distance in pixels. Hint: skimage.metrics.hausdorff_distance (handle empty masks)."""
    raise NotImplementedError


def ssim_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM between two binary masks. Hint: skimage.metrics.structural_similarity (data_range=1)."""
    raise NotImplementedError


def evaluate_methods(masks: Dict[str, np.ndarray], gt: np.ndarray) -> pd.DataFrame:
    """Table with one row per method: region size, DSC, Hausdorff, SSIM."""
    raise NotImplementedError
