"""Region growing from seed points."""
from collections import deque
from typing import Dict, Sequence, Tuple

import numpy as np

_OFFSETS = {
    4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    8: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
}


def region_growing(img: np.ndarray, seeds: Sequence[Tuple[int, int]], tol: float,
                   connectivity: int = 8) -> np.ndarray:
    """Grow a region from the seeds ((row, col) pixels).

    A neighbour joins the region if |I(p) - reference| <= tol, with reference = mean
    intensity of the seed pixels (fixed, so the result does not depend on visiting order).
    connectivity = 4 or 8 neighbours.
    """
    if connectivity not in _OFFSETS:
        raise ValueError("connectivity must be 4 or 8")
    img = np.asarray(img, dtype=float)
    rows, cols = img.shape
    seeds = [(int(r), int(c)) for r, c in seeds]
    for r, c in seeds:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"Seed {(r, c)} outside image of shape {img.shape}")
    reference = float(np.mean([img[r, c] for r, c in seeds]))

    mask = np.zeros(img.shape, dtype=bool)
    queue = deque()
    for r, c in seeds:
        if not mask[r, c]:
            mask[r, c] = True
            queue.append((r, c))
    offsets = _OFFSETS[connectivity]
    while queue:
        r, c = queue.popleft()
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not mask[nr, nc] \
                    and abs(img[nr, nc] - reference) <= tol:
                mask[nr, nc] = True
                queue.append((nr, nc))
    return mask


def compare_connectivity(img: np.ndarray, seeds: Sequence[Tuple[int, int]], tol: float) -> Dict[str, np.ndarray]:
    """Run the 4- and 8-connected versions with the same parameters and return both masks."""
    return {
        "4-connectivity": region_growing(img, seeds, tol, connectivity=4),
        "8-connectivity": region_growing(img, seeds, tol, connectivity=8),
    }
