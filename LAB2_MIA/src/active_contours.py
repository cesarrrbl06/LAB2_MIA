"""Classical "simple" active contours (snakes)."""
from typing import Tuple

import numpy as np
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.segmentation import active_contour


def circle_init(center: Tuple[float, float], radius: float, n_points: int = 400) -> np.ndarray:
    """Initial closed curve as an (n_points, 2) array of (row, col) coordinates."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r0, c0 = center
    return np.column_stack([r0 + radius * np.sin(t), c0 + radius * np.cos(t)])


def run_snake(img: np.ndarray, init: np.ndarray, alpha: float = 0.015, beta: float = 10.0,
              gamma: float = 0.001, w_line: float = 0.0, w_edge: float = 1.0, sigma: float = 3.0,
              max_num_iter: int = 2500) -> np.ndarray:
    """Evolve the snake on a Gaussian-smoothed image.

    alpha = elasticity, beta = rigidity (internal); w_line / w_edge = external forces;
    gamma = time step.
    """
    smoothed = gaussian(np.asarray(img, dtype=float), sigma=sigma, preserve_range=True)
    return active_contour(smoothed, np.asarray(init, dtype=float), alpha=alpha, beta=beta,
                          gamma=gamma, w_line=w_line, w_edge=w_edge,
                          max_num_iter=max_num_iter, boundary_condition="periodic")


def snake_to_mask(snake: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Filled binary mask from the final contour."""
    mask = np.zeros(shape, dtype=bool)
    rr, cc = polygon(snake[:, 0], snake[:, 1], shape=shape)
    mask[rr, cc] = True
    return mask
