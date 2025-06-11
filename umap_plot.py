import fcsparser
import argparse
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import logging
import os

# --- Logging Setup ---
log_file = os.path.join(os.path.dirname(__file__), "umap_run.log")
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# ----------------------

def run_umap_on_fcs(
    fcs_path: str = "20250520_ALL_Patient_BM_sort.fcs",
    output_image_path: str = "20250611_an_replication.png"
):
    """
    Parse an FCS file, perform UMAP on selected columns, and save a plot.

    Parameters:
        fcs_path (str): Path to the .fcs file.
        output_image_path (str): Path where the output image will be saved.
    """
    logger.info(f"Loading FCS file from: {fcs_path}")
    _, data = fcsparser.parse(fcs_path, reformat_meta=True)

    logger.info("Scaling selected columns (15 to 41)...")
    scaled = StandardScaler().fit_transform(data.iloc[:, 15:15+26])

    logger.info("Running UMAP...")
    start = time.time()
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.5)
    embedding = reducer.fit_transform(scaled)
    end = time.time()
    logger.info(f"UMAP took {end - start:.2f} seconds.")

    plt.figure(figsize=(10, 7))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=2)
    plt.title('UMAP Projection')
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    logger.info(f"Saved UMAP plot to: {output_image_path}")
    for var in range(15, 15+26):
        plot_umap_intensity(embedding, data.iloc[:, var].values, data.columns[var], f"20250611_umap_{data.columns[var]}.png")


def plot_umap_intensity(embeddings, color_var, color_name, output_image_path) -> None:
    norm = plt.Normalize(vmin=color_var.min(), vmax=color_var.max())
    colors = plt.cm.bwr(norm(color_var))  # blue to red colormap

    plt.figure(figsize=(10, 7))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=2, color=colors)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=plt.gca(), label=color_name)
    plt.title(f'UMAP Projection colored by {color_name} intensity')
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    logger.info(f"Saved UMAP plot to: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UMAP on FCS data and save plot.")
    parser.add_argument(
        "fcs_path", nargs="?", default="20250520_ALL_Patient_BM_sort.fcs",
        help="Path to the .fcs file (default: 20250520_ALL_Patient_BM_sort.fcs)"
    )
    parser.add_argument(
        "output_image_path", nargs="?", default="20250606_an_replication.png",
        help="Path to save the output PNG (default: 20250606_an_replication.png)"
    )
    args = parser.parse_args()

    run_umap_on_fcs(args.fcs_path, args.output_image_path)
