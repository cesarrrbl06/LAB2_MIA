"""Classical "simple" active contours (snakes)."""
from typing import Tuple

import numpy as np
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.transform import rescale


def ellipse_init(center: Tuple[float, float], radii: Tuple[float, float], n_points: int = 400) -> np.ndarray:
    """Initial closed ellipse as an (n_points, 2) array of (row, col) coordinates.

    radii = (radius along the rows, radius along the columns). An elongated structure such as
    the lung is better enclosed by an ellipse than by a circle.
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r0, c0 = center
    return np.column_stack([r0 + radii[0] * np.sin(t), c0 + radii[1] * np.cos(t)])


def circle_init(center: Tuple[float, float], radius: float, n_points: int = 400) -> np.ndarray:
    """Initial closed circle (an ellipse with equal radii)."""
    return ellipse_init(center, (radius, radius), n_points)


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


def snake_segmentation(img: np.ndarray, init: np.ndarray, alpha: float, beta: float,
                       sigma: float = 1.0, scale: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Segment with a snake and return (final contour, filled mask), both at the size of ``img``.

    ``init`` is the initial curve in the coordinates of ``img``. If ``scale`` > 1 the snake runs on
    the image reduced ``scale`` times: on a large image (e.g. the 1770 x 1000 px chest ROI) the edge
    force is too weak against the elasticity and the snake collapses, and it is also much slower.
    """
    small = rescale(img, 1 / scale, anti_aliasing=True) if scale > 1 else img
    snake = run_snake(small, np.asarray(init, dtype=float) / scale, alpha=alpha, beta=beta, sigma=sigma)
    snake = snake * scale                              # back to the pixel grid of img
    return snake, snake_to_mask(snake, img.shape)
