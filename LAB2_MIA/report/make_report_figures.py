"""Figures of the LAB 2 report (report/figures/), made from the best parameters found in the notebook.

Usage (from the project root):  python report/make_report_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import (active_contours, advanced_methods, export_seg, metrics, preprocessing,  # noqa: E402
                 region_growing, utils, watershed_seg)

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "legend.fontsize": 9})

# ----------------------------- data and best parameters (notebook) ----------------------------- #
CASES = {
    "brain": dict(file="Brain_MRI_131749_T1.png", gt="SegmentationTumor.nrrd", roi=(120, 205, 125, 215),
                  seeds=[(162, 168)], T=0.143, tol=0.09, conn=4, ws_grad="sobel", ws_sigma=4,
                  snake=dict(center=(161, 169), radii=(16, 16), scale=1, sigma=1, n_points=100, alpha=0.005, beta=10.0),
                  cv=dict(mu=0.05, lambda1=10)),
    "chest": dict(file="Chest_Xray_PA_3-8-2010.png", gt="SegmentationLung.nrrd", roi=(140, 1910, 130, 1130),
                  seeds=[(700, 600), (1000, 550), (1300, 700)], T=0.430, tol=0.21, conn=4,
                  ws_grad="morphological", ws_sigma=8,
                  snake=dict(center=(1025, 630), radii=(840, 480), scale=8, sigma=2, n_points=400, alpha=0.001, beta=10.0),
                  cv=dict(mu=0.5, lambda1=2)),
}
NAMES = {"brain": "Brain MRI", "chest": "Chest X-ray"}

data = {}
for key, c in CASES.items():
    img = utils.load_image(c["file"])
    gt_full = export_seg.read_mask(utils.DATA_GT / c["gt"], reference_shape=img.shape)
    roi = utils.crop_roi(img, c["roi"])
    seeds = utils.to_roi_coords(c["seeds"], c["roi"])
    markers = watershed_seg.build_markers(roi.shape, seeds, [], fg_radius=3)
    markers[[0, -1], :] = watershed_seg.BACKGROUND
    markers[:, [0, -1]] = watershed_seg.BACKGROUND
    data[key] = dict(img=img, roi=roi, gt=utils.crop_roi(gt_full, c["roi"]), seeds=seeds, markers=markers)


def overlay(ax, img, mask=None, gt=None, title="", color="red"):
    ax.imshow(img, cmap="gray")
    if gt is not None:
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1)
    if mask is not None and mask.any():
        ax.contour(mask, levels=[0.5], colors=color, linewidths=1.2)
    ax.set_title(title)
    ax.axis("off")


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------- Fig 1: ROIs and references --------------------------------- #
fig, axes = plt.subplots(1, 4, figsize=(13, 4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.25, 0.6]})
for (ax_full, ax_roi), key in zip([axes[:2], axes[2:]], CASES):
    d, (y0, y1, x0, x1) = data[key], CASES[key]["roi"]
    ax_full.imshow(d["img"], cmap="gray")
    ax_full.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.5))
    ax_full.set_title(f"{NAMES[key]} ({d['img'].shape[0]}×{d['img'].shape[1]}) + ROI")
    ax_full.axis("off")
    overlay(ax_roi, d["roi"], gt=d["gt"], title=f"ROI {d['roi'].shape[0]}×{d['roi'].shape[1]} + reference")
save(fig, "fig1_roi_reference.png")

# ---------------------------------- Fig 2: thresholding ---------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
for row, key in zip(axes, CASES):
    d = data[key]
    T_otsu = float(__import__("skimage.filters", fromlist=["x"]).threshold_otsu(d["roi"]))
    T_values = np.linspace(*np.percentile(d["roi"], [1, 99]), 100)
    dices = [metrics.dice(d["roi"] <= T, d["gt"]) for T in T_values]
    T_best = float(T_values[np.argmax(dices)])
    row[0].hist(d["roi"].ravel(), bins=128, color="gray")
    row[0].set(title=f"{NAMES[key]}: ROI histogram", xlabel="Intensity", ylabel="Pixels")
    row[1].plot(T_values, dices, color="black")
    row[1].set(title=f"{NAMES[key]}: Dice vs T", xlabel="Threshold T", ylabel="Dice", ylim=(0, 1.02))
    for ax in row[:2]:
        ax.axvline(T_otsu, color="tab:blue", linestyle="--", label=f"Otsu T = {T_otsu:.3f}")
        ax.axvline(T_best, color="tab:red", linestyle="--", label=f"Best T = {T_best:.3f}")
        ax.legend(fontsize=9)
    overlay(row[2], d["roi"], d["roi"] <= T_best, d["gt"], f"Best T: Dice = {max(dices):.3f}")
fig.tight_layout()
save(fig, "fig2_thresholding.png")

# ---------------------------------- Fig 3: region growing -------------------------------------- #
tols = np.round(np.arange(0.01, 0.41, 0.01), 2)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.3, 0.9, 1.3, 0.55]})
for (ax_c, ax_m), key in zip([axes[:2], axes[2:]], CASES):
    d, c = data[key], CASES[key]
    for conn, style in ((4, "-"), (8, "--")):
        ax_c.plot(tols, [metrics.dice(region_growing.region_growing(d["roi"], d["seeds"], t, conn), d["gt"])
                         for t in tols], style, label=f"{conn}-connectivity")
    ax_c.set(title=f"{NAMES[key]}: Dice vs tolerance", xlabel="Tolerance", ylabel="Dice", ylim=(0, 1.02))
    ax_c.legend(fontsize=9)
    mask = region_growing.region_growing(d["roi"], d["seeds"], c["tol"], c["conn"])
    overlay(ax_m, d["roi"], mask, d["gt"], f"tol = {c['tol']}: Dice = {metrics.dice(mask, d['gt']):.3f}")
    ax_m.plot([s[1] for s in d["seeds"]], [s[0] for s in d["seeds"]], "y+", markersize=8, markeredgewidth=2)
fig.tight_layout()
save(fig, "fig3_region_growing.png")

# ---------------------------------- Fig 4: watershed ------------------------------------------- #
ws_masks = {}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1, 0.6, 0.6]})
labels, n = watershed_seg.watershed_oversegmentation(data["brain"]["roi"])
axes[0].imshow(labels % 20, cmap="tab20")
axes[0].set_title(f"Brain MRI: no markers ({n} regions)")
axes[0].axis("off")
for ax, key in zip(axes[1:2].tolist() + axes[3:4].tolist(), CASES):
    d, c = data[key], CASES[key]
    smooth = preprocessing.smooth_gaussian(d["roi"], c["ws_sigma"])
    _, mask, _ = watershed_seg.run_watershed(smooth, d["markers"], method=c["ws_grad"])
    ws_masks[key] = mask
    grad = "Sobel" if c["ws_grad"] == "sobel" else "morph."
    overlay(ax, d["roi"], mask, d["gt"],
            f"{NAMES[key]}: {grad}, σ = {c['ws_sigma']}\nDice = {metrics.dice(mask, d['gt']):.3f}")
relief = watershed_seg.gradient_map(preprocessing.smooth_gaussian(data["chest"]["roi"], 8), method="morphological")
axes[2].imshow(relief, cmap="magma")
axes[2].set_title("Chest: relief (morph. gradient)")
axes[2].axis("off")
fig.tight_layout()
save(fig, "fig4_watershed.png")

# ---------------------------------- Fig 5: post-processing (chest) ----------------------------- #
d = data["chest"]
thr = d["roi"] <= CASES["chest"]["T"]
pairs = [("Thresholding", thr, 30), ("Watershed", ws_masks["chest"], 30)]
fig, axes = plt.subplots(1, 4, figsize=(9, 4.2))
for i, (name, mask, r) in enumerate(pairs):
    post = preprocessing.clean_mask(mask, opening_radius=r, hole_size=0, largest_only=True)
    overlay(axes[2 * i], d["roi"], mask, d["gt"], f"{name}\nDice = {metrics.dice(mask, d['gt']):.3f}")
    overlay(axes[2 * i + 1], d["roi"], post, d["gt"], f"{name} + post.\nDice = {metrics.dice(post, d['gt']):.3f}")
fig.tight_layout()
save(fig, "fig5_postprocessing.png")

# ---------------------------------- Fig 6: snakes ---------------------------------------------- #
INIT_TESTS = {
    "brain": {"Reference (r = 16)": ((161, 169), (16, 16)), "Tight (r = 10)": ((161, 169), (10, 10)),
              "Large (r = 25)": ((161, 169), (25, 25)), "Shifted 12 px": ((161, 181), (16, 16))},
    "chest": {"Reference": ((1025, 630), (840, 480)), "Smaller": ((1025, 630), (600, 350)),
              "Inside the lung": ((1025, 630), (400, 250)), "Shifted 130 px": ((1025, 500), (840, 480))},
}
fig, axes = plt.subplots(2, 4, figsize=(12, 9.2), gridspec_kw={"height_ratios": [1, 1.75]})
for row, key in zip(axes, CASES):
    d, s = data[key], CASES[key]["snake"]
    h, w = d["roi"].shape
    for ax, (name, (center, radii)) in zip(row, INIT_TESTS[key].items()):
        center = utils.to_roi_coords([center], CASES[key]["roi"])[0]
        init = active_contours.ellipse_init(center, radii, n_points=s["n_points"])
        snake, mask = active_contours.snake_segmentation(d["roi"], init, s["alpha"], s["beta"],
                                                         sigma=s["sigma"], scale=s["scale"])
        overlay(ax, d["roi"], gt=d["gt"], title=f"{name}\nDice = {metrics.dice(mask, d['gt']):.3f}")
        ax.plot(init[:, 1], init[:, 0], "--b", linewidth=1)
        ax.plot(snake[:, 1], snake[:, 0], "-r", linewidth=1.3)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
fig.tight_layout()
save(fig, "fig6_snakes.png")

# ---------------------------------- Fig 7: Chan-Vese ------------------------------------------- #
fig, axes = plt.subplots(1, 4, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1, 0.6, 0.6]})
tests = [("brain", (161, 169), (16, 16), "Reference ellipse"), ("brain", (161, 169), (25, 25), "Large ellipse (r = 25)"),
         ("chest", (1025, 630), (840, 480), "Reference ellipse"), ("chest", (1025, 630), (400, 250), "Inside the lung")]
for ax, (key, center, radii, name) in zip(axes, tests):
    d, s, cv = data[key], CASES[key]["snake"], CASES[key]["cv"]
    center = utils.to_roi_coords([center], CASES[key]["roi"])[0]
    mask = advanced_methods.chan_vese_segmentation(d["roi"], center, radii, mu=cv["mu"], lambda1=cv["lambda1"],
                                                   scale=s["scale"])
    overlay(ax, d["roi"], mask, d["gt"], f"{NAMES[key]}: {name}\nDice = {metrics.dice(mask, d['gt']):.3f}")
    init = active_contours.ellipse_init(center, radii, n_points=200)
    ax.plot(init[:, 1], init[:, 0], "--b", linewidth=1)
    ax.set_xlim(0, d["roi"].shape[1])
    ax.set_ylim(d["roi"].shape[0], 0)
fig.tight_layout()
save(fig, "fig7_chan_vese.png")

# ---------------------------------- Fig 8: noise robustness (tables of 3.4b) ------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, key in zip(axes, CASES):
    table = pd.read_csv(ROOT / "results" / "tables" / f"part3_noise_dice_{key}.csv", index_col=0)
    table.plot(ax=ax, marker="o", legend=(key == "brain"))
    ax.set(title=f"{NAMES[key]}: Dice vs noise", xlabel="Noise standard deviation σ", ylabel="Dice", ylim=(0, 1.02))
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=9)
fig.tight_layout()
save(fig, "fig8_noise.png")
