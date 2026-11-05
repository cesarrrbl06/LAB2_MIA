# LAB 2 — Medical Image Segmentation (MIA)

Medical Image Analysis · Biomedical Engineering · URJC 2026/2027.
Thresholding, region growing, watershed, active contours (snakes) and Chan-Vese, applied to a brain MRI
lesion and to the right lung of a chest X-ray, and evaluated against manual segmentations made in 3D Slicer.

## 📁 Project Structure

```text
LAB2_MIA/
│
├── raw_images/                        # Original images (the Lab 1 images)
│   ├── Brain_MRI_131749_T1.png        # T1-weighted Brain MRI
│   └── Chest_Xray_PA_3-8-2010.png     # Posteroanterior (PA) Chest X-ray
│
├── data/
│   └── ground_truth/                  # Manual segmentations from 3D Slicer (.nrrd) and binary masks (.png)
│
├── notebooks/
│   └── LAB_2.ipynb                    # Main notebook: all the experiments, tables and figures
│
├── src/                               # Python package used by the notebook
│   ├── utils.py                       # Paths, image loading, ROI handling, figure/table saving
│   ├── preprocessing.py               # Filtering, histogram modifications, morphological post-processing
│   ├── thresholding.py                # Otsu and manual thresholding
│   ├── region_growing.py              # Region growing (4- and 8-connectivity)
│   ├── watershed_seg.py               # Gradient map, markers, marker-controlled watershed
│   ├── active_contours.py             # Snakes (initial ellipse, evolution, mask)
│   ├── advanced_methods.py            # Part 3: Chan-Vese
│   ├── metrics.py                     # Dice, Hausdorff, SSIM and comparison table
│   └── export_seg.py                  # Reading of the 3D Slicer segmentations, binary masks
│
├── results/
│   ├── figures/                       # Figures saved by the notebook (300 DPI), also used in the report
│   └── tables/                        # Tables saved by the notebook (CSV)
│
├── report/
│   └── Lab_2_MIA.pdf                  # Assignment
│
├── main.py                            # Quick check: every method with its best parameters (~20 s)
├── requirements.txt
└── README.md
```

## 🚀 Usage

```bash
pip install -r requirements.txt
python main.py                     # metrics of every method with the best parameters
jupyter notebook notebooks/LAB_2.ipynb   # run all the cells (about 6 minutes)
```

## 📓 Notebook sections

* **Part 1** — 1.1 Images and ROIs · 1.2 Reference segmentations · 1.3 Thresholding · 1.4 Region growing ·
  1.5 Watershed · 1.6a Pre-processing · 1.6b Post-processing
* **Part 2** — 2.1 Snake model · 2.2 Initial contour · 2.3 Influence of α and β · 2.4 Influence of σ and scale ·
  2.5 Sensitivity to the initial contour · 2.6 Comparison with Part 1
* **Part 3** — 3.1 State of the art and Chan-Vese · 3.2 Initial level set · 3.3 Influence of μ and λ1 ·
  3.4 Robustness (initial contour and noise) · 3.5 Final comparison of all the methods

## 📦 Final Submission Format

`.zip` named `P2_MIA_BED_26_27_GroupX.zip` (e.g. `P2_MIA_BED_26_27_G2_1.zip`) containing:
1. Single PDF report (max. 10 pages) with the contribution of each member and the AI disclosure.
2. Source code (`src/`, `notebooks/`, `main.py`).
3. Images used (`raw_images/`, and `data/ground_truth/`).

Deadline: **Nov 5, 2026 23:55** (G1_x) / **Nov 6, 2026 23:55** (G2_x).
