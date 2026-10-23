"""Region growing from seed points.

SKELETON: implement each function (TODO).
"""
from typing import Dict, Sequence, Tuple

import numpy as np


def region_growing(img: np.ndarray, seeds: Sequence[Tuple[int, int]], tol: float,
                   connectivity: int = 8) -> np.ndarray:
    """Grow a region from the seeds ((row, col) pixels).

    A neighbour joins the region if |I(p) - reference| <= tol (decide the reference: seed
    mean or running region mean). connectivity = 4 or 8 neighbours.
    Hints: collections.deque (BFS) or skimage.segmentation.flood to cross-check your result.
    """
    raise NotImplementedError


def compare_connectivity(img: np.ndarray, seeds: Sequence[Tuple[int, int]], tol: float) -> Dict[str, np.ndarray]:
    """Run the 4- and 8-connected versions with the same parameters and return both masks."""
    raise NotImplementedError
