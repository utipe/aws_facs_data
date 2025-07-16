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
2. Install unzip:
    ```bash
    sudo apt update
    sudo apt install unzip 
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3.11-dev
    sudo apt install -y build-essential libffi-dev libssl-dev
3. Install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#getting-started-install-instructions
4. Download the required fcs file:
    ```bash
    aws s3 cp s3://kobe-u-cmsproject-flowjo/20250520_Inoue_patient_analysis/Tubes/20250520_ALL_Patient_BM_sort.fcs ./
5. Clone this repos
6. Create a virtual environment:
   ```bash
   cd aws_facs_data
   python3.11 -m venv umap
   source umap/bin/activate
   pip install .
---


## Usage

With default (fcs data should be placed in the same directory as umap_plot.py):

    ```bash
    python umap_plot.py


With custom path:

    ```bash
    python umap_plot.py path_fcs_data.fcs path_umap_image.png
