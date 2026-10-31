"""Region growing from seed points."""
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

_STRUCTURE = {4: ndi.generate_binary_structure(2, 1),   # cross: up, down, left, right
              8: ndi.generate_binary_structure(2, 2)}   # 3 x 3 square: also the diagonals


def region_growing(img: np.ndarray, seeds: Sequence[Tuple[int, int]], tol: float,
                   connectivity: int = 8) -> np.ndarray:
    """Grow a region from the seeds ((row, col) pixels).

    A pixel joins the region if |I(p) - reference| <= tol, with reference = mean intensity of
    the seeds (fixed, so the result does not depend on the visiting order), and if it is connected
    to a seed through pixels that also meet the criterion (connectivity = 4 or 8 neighbours).
    This is the same region a breadth-first search from the seeds would grow, computed with
    connected-component labelling (much faster on large ROIs).
    """
    if connectivity not in _STRUCTURE:
        raise ValueError("connectivity must be 4 or 8")
    img = np.asarray(img, dtype=float)
    seeds = [(int(r), int(c)) for r, c in seeds]
    for r, c in seeds:
        if not (0 <= r < img.shape[0] and 0 <= c < img.shape[1]):
            raise ValueError(f"Seed {(r, c)} outside image of shape {img.shape}")
    reference = float(np.mean([img[r, c] for r, c in seeds]))

    seed_mask = np.zeros(img.shape, dtype=bool)
    seed_mask[tuple(np.array(seeds).T)] = True
    candidates = np.abs(img - reference) <= tol             # homogeneity criterion
    labels, _ = ndi.label(candidates, structure=_STRUCTURE[connectivity])
    # regions that contain a seed or touch one (a seed is always part of the region)
    touched = ndi.binary_dilation(seed_mask, structure=_STRUCTURE[connectivity])
    grown = np.isin(labels, np.unique(labels[touched & candidates]))
    return grown | seed_mask


def compare_connectivity(img: np.ndarray, seeds: Sequence[Tuple[int, int]], tol: float) -> Dict[str, np.ndarray]:
    """Run the 4- and 8-connected versions with the same parameters and return both masks."""
    return {
        "4-connectivity": region_growing(img, seeds, tol, connectivity=4),
        "8-connectivity": region_growing(img, seeds, tol, connectivity=8),
    }
