"""End-to-end check of LAB 2: runs every method with the best parameters found in the notebook
on both ROIs and prints the metrics against the manual reference (it takes about 20 seconds).

Usage (from the project root):  python main.py
"""
import pandas as pd

from src import (active_contours, advanced_methods, export_seg, metrics, preprocessing, region_growing,
                 thresholding, utils, watershed_seg)

# Image, reference, ROI, seeds and best parameters of each structure (see notebooks/LAB_2.ipynb)
CASES = {
    "brain": {"file": "Brain_MRI_131749_T1.png", "gt": "SegmentationTumor.nrrd", "roi": (120, 205, 125, 215),
              "seeds": [(162, 168)], "T": 0.143, "tol": 0.09, "ws": ("sobel", 4),
              "snake": {"center": (161, 169), "radii": (16, 16), "scale": 1, "sigma": 1, "n_points": 100,
                        "alpha": 0.005, "beta": 10.0},
              "cv": {"mu": 0.05, "lambda1": 10}},
    "chest": {"file": "Chest_Xray_PA_3-8-2010.png", "gt": "SegmentationLung.nrrd", "roi": (140, 1910, 130, 1130),
              "seeds": [(700, 600), (1000, 550), (1300, 700)], "T": 0.430, "tol": 0.21, "ws": ("morphological", 8),
              "snake": {"center": (1025, 630), "radii": (840, 480), "scale": 8, "sigma": 2, "n_points": 400,
                        "alpha": 0.001, "beta": 10.0},
              "cv": {"mu": 0.5, "lambda1": 2}},
}


def main() -> None:
    utils.set_seed(42)
    for key, c in CASES.items():
        img = utils.load_image(c["file"])
        roi = utils.crop_roi(img, c["roi"])
        gt = utils.crop_roi(export_seg.read_mask(utils.DATA_GT / c["gt"], reference_shape=img.shape), c["roi"])
        seeds = utils.to_roi_coords(c["seeds"], c["roi"])
        center = utils.to_roi_coords([c["snake"]["center"]], c["roi"])[0]
        s = c["snake"]

        # watershed: object markers on the seeds, background marker on the frame of the ROI
        markers = watershed_seg.build_markers(roi.shape, seeds, [], fg_radius=3)
        markers[[0, -1], :] = watershed_seg.BACKGROUND
        markers[:, [0, -1]] = watershed_seg.BACKGROUND
        gradient, sigma = c["ws"]

        init = active_contours.ellipse_init(center, s["radii"], n_points=s["n_points"])
        masks = {
            "Thresholding": thresholding.apply_threshold(roi, high=c["T"]),
            "Region growing": region_growing.region_growing(roi, seeds, c["tol"], connectivity=4),
            "Watershed": watershed_seg.run_watershed(preprocessing.smooth_gaussian(roi, sigma), markers,
                                                     method=gradient)[1],
            "Snakes": active_contours.snake_segmentation(roi, init, s["alpha"], s["beta"], sigma=s["sigma"],
                                                         scale=s["scale"])[1],
            "Chan-Vese": advanced_methods.chan_vese_segmentation(roi, center, s["radii"], mu=c["cv"]["mu"],
                                                                 lambda1=c["cv"]["lambda1"], scale=s["scale"]),
        }
        print(f"\n{key}: ROI {roi.shape[0]} x {roi.shape[1]} px")
        print(metrics.evaluate_methods(masks, gt).round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
