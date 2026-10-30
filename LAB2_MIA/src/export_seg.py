"""Ground-truth reading and .seg.nrrd export (3D Slicer compatible) with pynrrd."""
from pathlib import Path
from typing import Optional, Tuple

import nrrd
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread, imsave

from .utils import ROI


def read_mask(path: Path, reference_shape: Optional[Tuple[int, int]] = None,
              flip_rows: bool = False) -> np.ndarray:
    """Read a manual segmentation (.nrrd / .seg.nrrd from 3D Slicer or .png from Paint) as a 2-D bool mask.

    nrrd stores [x, y] -> [row, col]. 3D Slicer may export only the bounding box of the segment, with
    its position in ``space origin`` / ``space directions``. Pass ``reference_shape`` (shape of the
    original image) to place the mask in the pixel grid of that image using this geometry; it assumes
    the image was loaded in Slicer with the default geometry of a PNG (origin 0, spacing 1).
    Set ``flip_rows=True`` if the mask looks vertically mirrored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    if path.name.endswith(".nrrd"):
        data, header = nrrd.read(str(path))
        directions = np.asarray(header.get("space directions", np.eye(data.ndim)), dtype=float)
        kinds = [str(k).lower() for k in header.get("kinds", [])]
        if data.ndim == 4 or (kinds and kinds[0] in ("list", "vector")):
            data = (data > 0).any(axis=0)  # several layers -> union
            directions = directions[1:]    # first axis was not spatial
        data = data > 0
        if data.ndim == 3:  # [x, y, z] -> union along z (a 2-D image has a single slice)
            data = data.any(axis=-1)
        if reference_shape is None:
            mask = data.T
        else:
            origin = np.asarray(header.get("space origin", np.zeros(3)), dtype=float)[:2]
            i, j = np.nonzero(data)
            xy = origin + np.outer(i, directions[0, :2]) + np.outer(j, directions[1, :2])
            cols, rows = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
            n_rows, n_cols = reference_shape
            inside = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
            if not inside.all():
                raise ValueError(f"{path.name}: {np.sum(~inside)} mask pixels fall outside an image of "
                                 f"shape {reference_shape}; check the reference image used in Slicer")
            mask = np.zeros(reference_shape, dtype=bool)
            mask[rows, cols] = True
    else:
        img = np.squeeze(imread(str(path)))
        if img.ndim == 3:
            if img.shape[-1] == 4:
                img = rgba2rgb(img)
            img = rgb2gray(img)
        mask = img > (img.max() / 2 if img.max() > 0 else 0)
        if reference_shape is not None and mask.shape != tuple(reference_shape):
            raise ValueError(f"{path.name}: mask shape {mask.shape} != image shape {tuple(reference_shape)}")
    if flip_rows:
        mask = mask[::-1, :]
    return np.ascontiguousarray(mask, dtype=bool)


def write_mask_png(mask: np.ndarray, path: Path) -> Path:
    """Save a binary mask as an 8-bit PNG (0 = background, 255 = structure)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(str(path), np.asarray(mask, dtype=bool).astype(np.uint8) * 255, check_contrast=False)
    return path


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
