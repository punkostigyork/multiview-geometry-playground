# 🏛️ Multiview Geometry & 3D Reconstruction Playground

A from-scratch Computer Vision engine that recovers 3D structure from 2D motion. This project bypasses "black-box" libraries to implement the fundamental linear algebra and geometric constraints that power modern SLAM, SfM, and Neural Rendering (NeRF) pipelines.

![Temple Reconstruction](assets/reconstruction_3D.gif)

## 🎯 The Core Problem
How do we turn a series of flat images into a metric 3D model? This repository solves this by implementing a full Structure-from-Motion (SfM) pipeline, addressing the three pillars of 3D vision: feature correspondence, outlier rejection, and geometric triangulation.

## 📂 Project Structure

```text
.
├── LICENSE                          # MIT Legal permissions
├── README.md                         # Project documentation and setup
├── assets
│   └── reconstruction_3D.gif         # 360-degree rotation demo
├── data                              # Source images (Middlebury Temple)
├── environment.yml                  # Conda environment configuration
├── notebooks
│   ├── notebook.ipynb                # Main reconstruction walkthrough
│   └── temple_reconstruction.ply     # Exported 3D point cloud file
├── requirements.txt                  # Pip dependencies
├── src
│   ├── camera.py                     # Pinhole model & ray generation
│   ├── features.py                   # SIFT detection & FLANN matching
│   ├── geometry.py                   # 8-point algorithm & triangulation
│   ├── main.py                       # CLI execution entry point
│   └── visualization.py              # Matplotlib & Plotly wrappers
└── tests
    └── test_geometry.py              # Math verification & unit tests

```

## 🛠️ Key Technical Implementations

### 1. Robust Epipolar Geometry
At the heart of the project is the **Normalized 8-Point Algorithm**. Because raw feature matches are often noisy, I implemented a custom **RANSAC (Random Sample Consensus)** wrapper for the Fundamental Matrix estimation.
* **The Engineering:** It allows the system to ignore "zombie" matches (incorrect correlations) and find the true geometric relationship between views.
* **The Result:** Achieved a mean reprojection error of **< 0.1 pixels**.

### 2. Sequential Triangulation & Color Recovery
The system iteratively processes image pairs. For every verified inlier, the 3D position is calculated using **Direct Linear Transform (DLT)**. 
* **Metric Accuracy:** Using the Middlebury dataset calibration, points are recovered in a real-world metric coordinate system.
* **Vertex Coloring:** RGB values are sampled from the source images to create a photorealistic sparse cloud.

### 3. NeRF Data Readiness
Going beyond traditional SfM, this engine includes a **Camera Ray Generator**. It calculates origin vectors and directions for every pixel, formatted specifically for training Neural Radiance Fields (NeRF).

---

## 🏗️ Project Architecture

| Component | Responsibility |
| :--- | :--- |
| **`src/camera.py`** | Pinhole model, World-to-Camera projection, and Ray casting. |
| **`src/geometry.py`** | 8-point algorithm, RANSAC, DLT Triangulation, and Error metrics. |
| **`src/features.py`** | SIFT-based keypoint detection and FLANN matching. |
| **`tests/`** | Geometric unit tests verifying ray normalization and projection math. |

---

## 📊 Performance Metrics
* **Feature Robustness:** Successfully handles up to **40% outlier ratio**.
* **Geometric Precision:** Final model verified via **Mean Reprojection Error (MRE)**.
* **Scalability:** Sequential loop architecture allows for N-image reconstruction.

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/punkostigyork/multiview-geometry-playground
cd multiview-geometry-playground
pip install -r requirements.txt
```

### Verification & Usage
**Run Geometric Tests:**
```bash
PYTHONPATH=. pytest -v
```
**Run Reconstruction:**
Open ```notebooks/notebook.ipynb``` to walk through the step-by-step pipeline from raw images to a .ply export.

---
*Developed as a deep-dive into the mathematical foundations of spatial computing and computer vision.*

---

## 📄 Cite

```bibtex
@misc{multiview-geometry-playground,
  author = {Györk Pünkösti},
  title = {Multiview Geometry & 3D Reconstruction Playground},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/punkostigyork/multiview-geometry-playground}
}
```

## Licence

MIT