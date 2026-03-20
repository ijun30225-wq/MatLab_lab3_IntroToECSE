# ECSE 1010 — Lab 3: Traffic Sign Recognition

MATLAB code for Lab 3 (Parts A & B) of ECSE 1010: Introduction to ECSE at RPI.

**Authors:** Nicholas Gladu, Jun Iguchi

---

## Lab 3A — Image Processing of Traffic Sign Photos

### Overview
Processes traffic sign images using classical image processing techniques to isolate red objects (stop signs) and extract shape and color features.

### Pipeline
1. **Data Import** — Load images from the `3A_traffic_sign_images` datastore.
2. **Preprocessing** — Convert to LAB color space and apply adaptive histogram equalization.
3. **Color Band Separation** — Split into R, G, B channels and plot histograms.
4. **Thresholding / Masking** — Build a red-only mask: high red, low green, low blue.
5. **Morphological Refinement** — Remove noise (`bwareaopen`), fill holes (`imfill`), smooth edges (`imclose`).
6. **Feature Extraction** — Compute circularity and mean RGB over the largest detected region.
7. **Edge Detection** — Apply Sobel edge detection to reveal sign shape.
8. **Classification** — Rule-based: uses mean red intensity and circularity to label as Stop Sign, Speed Limit Sign, or Pedestrian Sign.

### Requirements
- MATLAB + Image Processing Toolbox
- `3A_traffic_sign_images/` folder on your MATLAB path

### Usage
```matlab
% Set n to the image index you want to inspect, then run:
run('Lab_3A.m')
```

---

## Lab 3B — Machine Learning for Traffic Sign Recognition

### Overview
Uses the **K-Nearest Neighbors (KNN)** algorithm to classify road signs into three categories: Stop, Speed Limit, and Crosswalk.

### Pipeline
1. **Data Import** — Load training/test images; parse XML annotations via `parseAnnotations`.
2. **Feature Extraction** (`extractFeatures` function):
   - Grayscale conversion + adaptive binarization
   - Noise removal and hole filling
   - Largest connected component → bounding box
   - **Shape:** Circularity, Eccentricity (`regionprops`)
   - **Color:** Mean R/G/B, normalized ratios, red dominance score
3. **Normalization** — Z-score normalize features; save center/scale for test data.
4. **Training** — `fitcknn` with 4 neighbors, 3 classes.
5. **Evaluation** — Confusion chart + accuracy on test set.

### Results
Final model achieved **~63% accuracy** on the test set (60–70% bonus tier).

### Requirements
- MATLAB + Image Processing Toolbox + Statistics and Machine Learning Toolbox
- `3B_training_images/`, `3B_test_images/`, `3B_training_image_annotations/`, `3B_test_image_annotations/` on your MATLAB path
- `parseAnnotations.m` function file

### Usage
```matlab
% Make sure all data folders are on your MATLAB path, then run:
run('Lab_3B.m')
```

---

## Repository Structure

```
ecse1010-lab3/
├── README.md
├── .gitignore
├── Lab_3A.m          # Part A: classical image processing
└── Lab_3B.m          # Part B: KNN machine learning
```

> **Note:** Image datasets and annotation files are not included due to size.
> Download them from the course materials page and place them alongside the scripts.
