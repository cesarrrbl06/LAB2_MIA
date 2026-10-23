"""Ground-truth reading and .seg.nrrd export (3D Slicer compatible) with pynrrd.

SKELETON: implement each function (TODO).
"""
from pathlib import Path
from typing import Tuple

import numpy as np

from .utils import ROI


def read_mask(path: Path) -> np.ndarray:
    """Read a manual segmentation (.seg.nrrd from 3D Slicer or .png from Paint) as a 2-D bool mask.

    Hints: nrrd.read ([x, y] order -> transpose), np.squeeze. If the mask looks vertically
    mirrored, flip the rows (Slicer may store 2-D images upside down).
    """
    raise NotImplementedError


def embed_roi_mask(mask_roi: np.ndarray, full_shape: Tuple[int, int], roi: ROI) -> np.ndarray:
    """Place a ROI-sized mask back into an array with the original image dimensions."""
    raise NotImplementedError


def write_seg_nrrd(mask: np.ndarray, path: Path, segment_name: str = "ROI",
                   color: Tuple[float, float, float] = (0.9, 0.25, 0.25)) -> Path:
    """Write the mask as a .seg.nrrd with a proper header.

    Header fields to define: space, space directions, kinds, encoding, Segment0_ID,
    Segment0_Name, Segment0_Color, Segment0_LabelValue, Segment0_Layer, Segment0_Extent,
    Segmentation_MasterRepresentation. Hint: nrrd.write(..., index_order='F').
    """
    raise NotImplementedError
