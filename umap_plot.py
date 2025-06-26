import fcsparser
import argparse
import umap
import matplotlib.pyplot as plt
import time
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import gc
import os
from spade_density_downsample import spade_density_downsample_faiss


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
    cofactor: float = 150.0,
    downsample_levels: list[int] = [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2, 1],
):
    logger.info(f"Loading FCS file from: {fcs_path}")
    _, data = fcsparser.parse(fcs_path, reformat_meta=True)

    logger.info("Extracting relevant columns and converting to NumPy...")
    X_full = data.iloc[:, 15:15+26].to_numpy(dtype=np.float32)
    del data
    gc.collect()

    for pct in downsample_levels:
        logger.info(f"--- Processing downsample level: {pct}% ---")

        # Downsampling
        start = time.time()
        X_ds = spade_density_downsample_faiss(X_full, target_pctile=pct)
        logger.info(f"Downsampling completed in {time.time() - start:.2f} seconds.")
        if len(X_ds) == 0:
            logger.warning(f"No data left after downsampling at {pct}%. Skipping.")
            continue

        # Arcsinh transformation
        X_trans = np.arcsinh(X_ds / cofactor)

        # UMAP
        logger.info("Running UMAP...")
        start = time.time()
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.5)
        embedding = reducer.fit_transform(X_trans)
        logger.info(f"UMAP completed in {time.time() - start:.2f} seconds.")

        # Save embedding
        emb_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
        emb_df.to_csv(f"umap_embeddings_{pct}_sample.csv", index=False)
        del emb_df
        gc.collect()

        # Density estimation
        logger.info("Computing local density...")
        nn = NearestNeighbors(n_neighbors=50, algorithm='auto').fit(embedding)
        distances, _ = nn.kneighbors(embedding)
        density = 1.0 / (distances.mean(axis=1) + 1e-8)

        # Plot
        logger.info("Generating plot...")
        plt.figure(figsize=(10, 10))
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=density,
            cmap="viridis",
            s=2,
            alpha=0.5
        )
        plt.colorbar(label="Local Density")
        plt.title(f"UMAP with Density Coloring ({pct}%)", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"UMAP_{pct}_sample.png", dpi=300)
        plt.close()

        # Clean up
        del X_ds, X_trans, embedding, distances, density
        gc.collect()

        logger.info(f"Completed {pct}% downsample.")



def plot_umap_intensity(embeddings, color_var, color_name, output_image_path) -> None:
    percentiles = np.percentile(color_var, np.linspace(0, 100, len(color_var)))
    norm = plt.Normalize(vmin=0, vmax=100)
    colors = plt.cm.bwr(norm(percentiles))

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
