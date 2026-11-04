"""Part 3 - additional segmentation method: Chan-Vese (active contours without edges)."""
from typing import Optional, Tuple

import numpy as np
from skimage.segmentation import chan_vese
from skimage.transform import rescale, resize


def ellipse_level_set(shape: Tuple[int, int], center: Tuple[float, float],
                      radii: Tuple[float, float]) -> np.ndarray:
    """Initial level set: +1 inside the ellipse, -1 outside (zero level = the ellipse).

    A binary (+1 / -1) level set is used instead of a signed distance: the Chan-Vese update is
    weighted by a smoothed delta of the level set, which is almost zero far from the curve when the
    values are large distances. With +1 / -1 every pixel can change phase from the first
    iteration, so the result depends much less on the size of the initial ellipse.
    """
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    inside = ((rows - center[0]) / radii[0]) ** 2 + ((cols - center[1]) / radii[1]) ** 2 < 1
    return np.where(inside, 1.0, -1.0)


def chan_vese_segmentation(img: np.ndarray, center: Tuple[float, float],
                           radii: Optional[Tuple[float, float]] = None, mu: float = 0.25,
                           lambda1: float = 1.0, lambda2: float = 1.0, max_num_iter: int = 500,
                           scale: int = 1) -> np.ndarray:
    """Chan-Vese segmentation; returns the object mask at the size of ``img``.

    center = (row, col) pixel inside the structure. Chan-Vese returns two phases without a label;
             the object is the phase that contains ``center``.
    radii  = radii of the initial ellipse around ``center`` (as in the snake). None -> scikit-image
             "checkerboard" initialisation (many small circles over the whole image).
    lambda1 weights the fit of the region where the level set is positive, i.e. the inside of the
    initial ellipse (the object); lambda2 weights the outside.
    scale  = downsampling factor applied before segmenting (as in the snake), to save time.
    """
    small = rescale(img, 1 / scale, anti_aliasing=True) if scale > 1 else np.asarray(img, dtype=float)
    r0, c0 = center[0] / scale, center[1] / scale          # center in the reduced image
    if radii is None:
        init_level_set = "checkerboard"
    else:
        init_level_set = ellipse_level_set(small.shape, (r0, c0), (radii[0] / scale, radii[1] / scale))

    seg = chan_vese(small, mu=mu, lambda1=lambda1, lambda2=lambda2, max_num_iter=max_num_iter,
                    init_level_set=init_level_set)
    r0 = int(np.clip(round(r0), 0, small.shape[0] - 1))
    c0 = int(np.clip(round(c0), 0, small.shape[1] - 1))
    if not seg[r0, c0]:          # keep the phase that contains the structure
        seg = ~seg
    if scale > 1:                # back to the pixel grid of img
        seg = resize(seg.astype(float), img.shape) > 0.5
    return seg
