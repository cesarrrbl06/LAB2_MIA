# LAB 2 — Medical Image Segmentation (MIA)

Medical Image Analysis · Biomedical Engineering · URJC 2026/2027.
Thresholding, region growing, watershed, active contours (snakes) and one additional method.

> **Status:** project skeleton. `src/` contains function signatures with `NotImplementedError`,
> and `notebooks/LAB_2.ipynb` contains the section outline only (no code).

## 📁 Project Structure

```text
LAB2_MIA/
│
├── data/
│   ├── raw/                           # Original images (reuse the Lab 1 images)
│   │   ├── Brain_MRI_131749_T1.png    # T1-weighted Brain MRI
│   │   └── Chest_Xray_PA_3-8-2010.png # Posteroanterior (PA) Chest X-ray
│   └── ground_truth/                  # Manual segmentations (3D Slicer .seg.nrrd or Paint .png)
│
├── notebooks/
│   └── LAB_2.ipynb                    # Main notebook (11 sections, English)
│
├── src/                               # Modular Python package
│   ├── __init__.py
│   ├── utils.py                       # Paths, image loading, ROI handling, figure/table saving
│   ├── preprocessing.py               # Filtering, histogram modifications, morphological post-processing
│   ├── thresholding.py                # Otsu / multi-Otsu / manual thresholding
│   ├── region_growing.py              # Region growing (4- and 8-connectivity)
│   ├── watershed_seg.py               # Gradient map, markers, marker-controlled watershed
│   ├── active_contours.py             # Snakes (initial curve, evolution, mask)
│   ├── advanced_methods.py            # Part 3: additional method of your choice
│   ├── metrics.py                     # Dice, Hausdorff, SSIM and comparison table
│   └── export_seg.py                  # Read ground truth, embed ROI, write .seg.nrrd
│
├── results/
│   ├── figures/                       # Exported plots for the report (300 DPI)
│   ├── tables/                        # CSVs (e.g. 02_segmentation_metrics_summary.csv)
│   └── segmentations/                 # Final .seg.nrrd files
│
├── report/
│   ├── Lab_2_MIA.pdf                  # Official assignment prompt
│   └── (Final Report PDF)             # Final report (max. 10 pages)
│
├── main.py                            # End-to-end pipeline verification script
├── requirements.txt                   # Environment dependencies
└── README.md
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📓 Notebook sections

1. Environment Setup & Project Verification
2. Clinical Data Loading & ROI Definition
3. Thresholding
4. Region Growing
5. Watershed
6. Active Contours (Snakes)
7. Pre- and Post-Processing *(added: required by the assignment)*
8. Part 3 — Additional Method *(added: required by the assignment)*
9. Ground Truth Integration & Quantitative Evaluation (DSC, Hausdorff, SSIM)
10. Exporting to `.seg.nrrd`
11. Team Contributions & AI Disclosure *(added: required by the assignment)*

## 📦 Final Submission Format

`.zip` named `P2_MIA_BED_26_27_GroupX.zip` (e.g. `P2_MIA_BED_26_27_G2_1.zip`) containing:
1. Single PDF report (max. 10 pages) with the contribution of each member and the AI disclosure.
2. Source code (`src/`, `notebooks/`, `main.py`).
3. Images used (`data/raw/`, and `data/ground_truth/`).

Deadline: **Nov 5, 2026 23:55** (G1_x) / **Nov 6, 2026 23:55** (G2_x).
