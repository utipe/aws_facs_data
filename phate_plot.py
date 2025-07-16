import fcsparser
import argparse
import phate
import matplotlib.pyplot as plt
import time
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import gc
import os
from spade_density_downsample import spade_density_downsample_faiss
from scipy import stats


# --- Logging Setup ---
log_file = os.path.join(os.path.dirname(__file__), "phate_run.log")
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# ----------------------


def run_phate_on_fcs(
    fcs_path: str = "20250520_ALL_Patient_BM_sort.fcs",
    cofactor: float = 150.0,
    pct: int = 30,
):
    logger.info(f"Loading FCS file from: {fcs_path}")
    _, data = fcsparser.parse(fcs_path, reformat_meta=True)

    logger.info("Extracting relevant columns and converting to NumPy...")
    X_full = data.iloc[:, 15:15+26].to_numpy(dtype=np.float32)

    logger.info(f"--- Processing downsample level: {pct}% ---")

    # Downsampling
    start = time.time()
    X_ds = spade_density_downsample_faiss(X_full, target_pctile=pct)
    logger.info(f"Downsampling completed in {time.time() - start:.2f} seconds.")

    # Arcsinh transformation
    X_trans = np.arcsinh(X_ds / cofactor)

        # PHATE
    logger.info("Running PHATE...")
    start = time.time()
    reducer = phate.PHATE(n_components=3, n_jobs=-1, verbose=3)
    embedding = reducer.fit_transform(X_trans)
    logger.info(f"UMAP completed in {time.time() - start:.2f} seconds.")

    # Save embedding
    emb_df = pd.DataFrame(embedding, columns=["phate_1", "phate_2", "phate_3"])
    emb_df.to_csv(f"phate_embeddings_{pct}_sample.csv", index=False)
    del emb_df
    gc.collect()

    # Density estimation
    logger.info("Computing local density...")
    nn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(embedding)
    distances, _ = nn.kneighbors(embedding)
    density = 1.0 / (distances.mean(axis=1) + 1e-8)

    # Plot
    logger.info("Generating plot...")
    plt.figure(figsize=(10, 10))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 2],
        c=density,
        cmap="bwr",
        s=2,
        alpha=0.5
    )
    plt.colorbar(label="Local Density")
    plt.title(f"PHATE with Density Coloring ({pct}%)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"PHATE_{pct}_sample1.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.scatter(
        embedding[:, 1],
        embedding[:, 2],
        c=density,
        cmap="bwr",
        s=2,
        alpha=0.5
    )
    plt.colorbar(label="Local Density")
    plt.title(f"PHATE with Density Coloring ({pct}%)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"PHATE_{pct}_sample2.png", dpi=300)
    plt.close()

    # Clean up
    del X_ds, X_trans, embedding, distances, density
    gc.collect()

    logger.info(f"Completed {pct}% downsample.")

    for var in range(15, 15+26):
        print(f"start mapping for {data.columns[var]}")
        plot_phate_intensity(embedding, data.iloc[:, var].values, data.columns[var], f"20250611_phate_{data.columns[var]}")



def plot_phate_intensity(embeddings, color_var, color_name, output_image_path) -> None:
    percentiles = [stats.percentileofscore(color_var, value, kind='rank') for value in color_var]
    norm = plt.Normalize(vmin=0, vmax=100)
    colors = plt.cm.bwr(norm(percentiles))

    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 2], s=2, color=colors)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=plt.gca(), label=color_name)
    plt.title(f'PHATE Projection colored by {color_name} intensity')
    plt.savefig(f"{output_image_path}1.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 1], embeddings[:, 1], s=2, color=colors)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=plt.gca(), label=color_name)
    plt.title(f'PHATE Projection colored by {color_name} intensity')
    plt.savefig(f"{output_image_path}2.png", dpi=300)
    plt.close()
    logger.info(f"Saved PHATE plot to: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PHATE on FCS data and save plot.")
    parser.add_argument(
        "fcs_path", nargs="?", default="20250520_ALL_Patient_BM_sort.fcs",
        help="Path to the .fcs file (default: 20250520_ALL_Patient_BM_sort.fcs)"
    )
    parser.add_argument(
        "output_image_path", nargs="?", default="20250716_an_replication.png",
        help="Path to save the output PNG (default: 20250716_an_replication.png)"
    )
    args = parser.parse_args()

    run_phate_on_fcs(args.fcs_path, args.output_image_path)
