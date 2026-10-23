"""End-to-end pipeline verification for LAB 2 (segmentation).

SKELETON: once the modules in src/ are implemented, use this script to check that
everything loads and runs on one image before running the notebook.

Usage:  python main.py
"""
from src import (active_contours, advanced_methods, export_seg, metrics,  # noqa: F401
                 preprocessing, region_growing, thresholding, utils, watershed_seg)


def main() -> None:
    # TODO 1. utils.set_seed / utils.ensure_dirs
    # TODO 2. load one image from data/raw and crop a ROI
    # TODO 3. run thresholding, region growing, watershed, snakes and the Part 3 method
    # TODO 4. compute the metrics table against a ground truth (if available)
    # TODO 5. embed the mask in the original dimensions and write/read a .seg.nrrd (temp folder)
    raise NotImplementedError


if __name__ == "__main__":
    main()
