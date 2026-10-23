"""Part 3 - one additional segmentation method of your choice.

SKELETON: keep ONLY the method you choose (Chan-Vese and Random Walker are suggestions).
"""
from typing import Optional, Sequence, Tuple

import numpy as np


def chan_vese_segmentation(img: np.ndarray, mu: float = 0.25, lambda1: float = 1.0, lambda2: float = 1.0,
                           max_num_iter: int = 200, init_center: Optional[Tuple[int, int]] = None,
                           init_radius: Optional[float] = None) -> np.ndarray:
    """Chan-Vese active contours without edges. Hint: skimage.segmentation.chan_vese.

    Note: it returns two unlabeled phases; decide which one is the object.
    """
    raise NotImplementedError


def random_walker_segmentation(img: np.ndarray, fg_points: Sequence[Tuple[int, int]],
                               bg_points: Sequence[Tuple[int, int]], beta: float = 130.0) -> np.ndarray:
    """Seeded Random Walker. Hint: skimage.segmentation.random_walker."""
    raise NotImplementedError
