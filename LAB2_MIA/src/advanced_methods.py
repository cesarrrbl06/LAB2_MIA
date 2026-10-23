"""Part 3 - one additional segmentation method of your choice.

Keep ONLY the method you choose (Chan-Vese and Random Walker are suggestions).
"""
from typing import Optional, Sequence, Tuple

import numpy as np
from skimage.segmentation import chan_vese, random_walker


def chan_vese_segmentation(img: np.ndarray, mu: float = 0.25, lambda1: float = 1.0, lambda2: float = 1.0,
                           max_num_iter: int = 200, init_center: Optional[Tuple[int, int]] = None,
                           init_radius: Optional[float] = None) -> np.ndarray:
    """Chan-Vese active contours without edges.

    Chan-Vese returns two unlabeled phases. The object is the phase that contains
    ``init_center`` (if given) or, otherwise, the phase that contains the image centre.
    If ``init_center`` / ``init_radius`` are given, the level set starts as that circle.
    """
    img = np.asarray(img, dtype=float)
    rows, cols = img.shape
    if init_center is not None or init_radius is not None:
        r0, c0 = init_center if init_center is not None else (rows / 2, cols / 2)
        radius = init_radius if init_radius is not None else min(rows, cols) / 4
        yy, xx = np.mgrid[:rows, :cols]
        init_level_set = radius - np.sqrt((yy - r0) ** 2 + (xx - c0) ** 2)
    else:
        init_level_set = "checkerboard"
    seg = chan_vese(img, mu=mu, lambda1=lambda1, lambda2=lambda2, max_num_iter=max_num_iter,
                    init_level_set=init_level_set)
    r_ref, c_ref = init_center if init_center is not None else (rows // 2, cols // 2)
    r_ref = int(np.clip(round(r_ref), 0, rows - 1))
    c_ref = int(np.clip(round(c_ref), 0, cols - 1))
    return seg if seg[r_ref, c_ref] else ~seg


def random_walker_segmentation(img: np.ndarray, fg_points: Sequence[Tuple[int, int]],
                               bg_points: Sequence[Tuple[int, int]], beta: float = 130.0) -> np.ndarray:
    """Seeded Random Walker (label 1 = background, 2 = foreground)."""
    img = np.asarray(img, dtype=float)
    labels = np.zeros(img.shape, dtype=np.int32)
    for r, c in bg_points:
        labels[int(r), int(c)] = 1
    for r, c in fg_points:
        labels[int(r), int(c)] = 2
    result = random_walker(img, labels, beta=beta, mode="cg_j")
    return result == 2
