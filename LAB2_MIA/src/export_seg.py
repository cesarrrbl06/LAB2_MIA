"""Ground-truth reading and .seg.nrrd export (3D Slicer compatible) with pynrrd."""
from pathlib import Path
from typing import Tuple

import nrrd
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread

from .utils import ROI


def read_mask(path: Path, flip_rows: bool = False) -> np.ndarray:
    """Read a manual segmentation (.seg.nrrd from 3D Slicer or .png from Paint) as a 2-D bool mask.

    nrrd stores [x, y] -> transposed to [row, col]. Set ``flip_rows=True`` if the mask looks
    vertically mirrored (Slicer may store 2-D images upside down).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    if path.name.endswith(".nrrd"):
        data, header = nrrd.read(str(path))
        kinds = [str(k).lower() for k in header.get("kinds", [])]
        if data.ndim == 4 or (kinds and kinds[0] in ("list", "vector")):
            data = (data > 0).any(axis=0)  # several layers -> union
        data = np.squeeze(data)
        if data.ndim == 3:  # [x, y, z] with several slices -> union along z
            data = data.any(axis=-1)
        mask = data.T > 0
    else:
        img = np.squeeze(imread(str(path)))
        if img.ndim == 3:
            if img.shape[-1] == 4:
                img = rgba2rgb(img)
            img = rgb2gray(img)
        mask = img > (img.max() / 2 if img.max() > 0 else 0)
    if flip_rows:
        mask = mask[::-1, :]
    return np.ascontiguousarray(mask, dtype=bool)


def embed_roi_mask(mask_roi: np.ndarray, full_shape: Tuple[int, int], roi: ROI) -> np.ndarray:
    """Place a ROI-sized mask back into an array with the original image dimensions."""
    ymin, ymax, xmin, xmax = roi
    mask_roi = np.asarray(mask_roi, dtype=bool)
    if mask_roi.shape != (ymax - ymin, xmax - xmin):
        raise ValueError(f"ROI mask shape {mask_roi.shape} does not match ROI {roi}")
    full = np.zeros(full_shape, dtype=bool)
    full[ymin:ymax, xmin:xmax] = mask_roi
    return full


def write_seg_nrrd(mask: np.ndarray, path: Path, segment_name: str = "ROI",
                   color: Tuple[float, float, float] = (0.9, 0.25, 0.25)) -> Path:
    """Write the mask as a .seg.nrrd with a proper header (3D Slicer segmentation)."""
    path = Path(path)
    if not path.name.endswith(".seg.nrrd"):
        path = path.with_name(path.name.split(".")[0] + ".seg.nrrd")
    path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(mask, dtype=bool)
    # numpy [row, col] -> nrrd [x, y, z] with a single slice
    data = mask.T.astype(np.uint8)[:, :, np.newaxis]
    nx, ny, nz = data.shape
    header = {
        "type": "unsigned char",
        "dimension": 3,
        "space": "left-posterior-superior",
        "space directions": np.eye(3),
        "space origin": np.zeros(3),
        "kinds": ["domain", "domain", "domain"],
        "encoding": "gzip",
        "Segment0_ID": "Segment_1",
        "Segment0_Name": segment_name,
        "Segment0_Color": " ".join(f"{c:g}" for c in color),
        "Segment0_LabelValue": "1",
        "Segment0_Layer": "0",
        "Segment0_Extent": f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}",
        "Segmentation_MasterRepresentation": "Binary labelmap",
        "Segmentation_ContainedRepresentationNames": "Binary labelmap|",
        "Segmentation_ReferenceImageExtentOffset": "0 0 0",
    }
    nrrd.write(str(path), data, header, index_order="F")
    return path
