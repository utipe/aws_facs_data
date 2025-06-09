# UMAP FCS Plotter

This script parses `.fcs` (Flow Cytometry Standard) files, scales relevant data columns, runs UMAP dimensionality reduction, and plots the result using Matplotlib.

---

## 📦 Dependencies

This project uses the following Python libraries:

- `fcsparser`
- `pandas`
- `umap-learn`
- `matplotlib`
- `scikit-learn`

These are listed in the `pyproject.toml`.

---

## 🚀 Setup

1. Make sure you're using **Python 3.8 or later**.
2. Create a virtual environment:
   ```bash
   python -m venv umap
   source umap/bin/activate
---

## Usage

With default (fcs data should be placed in the same directory as umap_plot.py):

    ```bash
    python umap_plot.py


With custom path:

    ```bash
    python umap_plot.py path_fcs_data.fcs path_umap_image.png
