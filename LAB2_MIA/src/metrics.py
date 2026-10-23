"""Segmentation metrics against a ground-truth mask."""
from typing import Dict

import numpy as np
import pandas as pd
from skimage.metrics import hausdorff_distance, structural_similarity


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice Similarity Coefficient: 2|A∩B| / (|A| + |B|)."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0  # both empty -> perfect agreement
    return float(2.0 * np.logical_and(pred, gt).sum() / total)


def hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hausdorff distance in pixels (0 if both masks are empty, inf if only one is)."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")
    return float(hausdorff_distance(pred, gt))


def ssim_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM between two binary masks (data_range=1)."""
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    win_size = min(7, *pred.shape)
    if win_size % 2 == 0:
        win_size -= 1
    return float(structural_similarity(pred, gt, data_range=1.0, win_size=win_size))


def evaluate_methods(masks: Dict[str, np.ndarray], gt: np.ndarray) -> pd.DataFrame:
    """Table with one row per method: region size, DSC, Hausdorff, SSIM."""
    gt = np.asarray(gt, dtype=bool)
    rows = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise ValueError(f"{name}: mask shape {mask.shape} != ground truth shape {gt.shape}")
        rows.append({"Method": name, "Region size (px)": int(mask.sum()), "DSC": dice(mask, gt),
                     "Hausdorff (px)": hausdorff(mask, gt), "SSIM": ssim_binary(mask, gt)})
    return pd.DataFrame(rows).set_index("Method")
