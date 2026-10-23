"""Classical "simple" active contours (snakes).

SKELETON: implement each function (TODO).
"""
from typing import Tuple

import numpy as np


def circle_init(center: Tuple[float, float], radius: float, n_points: int = 400) -> np.ndarray:
    """Initial closed curve as an (n_points, 2) array of (row, col) coordinates."""
    raise NotImplementedError


def run_snake(img: np.ndarray, init: np.ndarray, alpha: float = 0.015, beta: float = 10.0,
              gamma: float = 0.001, w_line: float = 0.0, w_edge: float = 1.0, sigma: float = 3.0,
              max_num_iter: int = 2500) -> np.ndarray:
    """Evolve the snake on a Gaussian-smoothed image.

    alpha = elasticity, beta = rigidity (internal); w_line / w_edge = external forces;
    gamma = time step. Hints: skimage.filters.gaussian, skimage.segmentation.active_contour.
    """
    raise NotImplementedError


def snake_to_mask(snake: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Filled binary mask from the final contour. Hint: skimage.draw.polygon."""
    raise NotImplementedError
