import fcsparser
import argparse
import pacmap
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
log_file = os.path.join(os.path.dirname(__file__), "pacmap_run.log")
logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# ----------------------


def run_pacmap_on_fcs(
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

    # UMAP
    logger.info("Running PACMAP...")
    start = time.time()
    reducer = pacmap.PaCMAP(n_components=2)
    embedding = reducer.fit_transform(X_trans)
    logger.info(f"PACMAP completed in {time.time() - start:.2f} seconds.")

    # Save embedding
    emb_df = pd.DataFrame(embedding, columns=["PACMAP_1", "PACMAP_2"])
    emb_df.to_csv(f"pacmap_embeddings_{pct}_sample.csv", index=False)
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
        embedding[:, 1],
        c=density,
        cmap="bwr",
        s=2,
        alpha=0.5
    )
    plt.colorbar(label="Local Density")
    plt.title(f"PACMAP with Density Coloring ({pct}%)", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"PACMAP_{pct}_sample.png", dpi=300)
    plt.close()

    # Clean up
    del X_ds, X_trans, embedding, distances, density
    gc.collect()

    logger.info(f"Completed {pct}% downsample.")

    for var in range(15, 15+26):
        print(f"start mapping for {data.columns[var]}")
        plot_pacmap_intensity(embedding, data.iloc[:, var].values, data.columns[var], f"20250611_pacmap_{data.columns[var]}.png")



def plot_pacmap_intensity(embeddings, color_var, color_name, output_image_path) -> None:
    percentiles = [stats.percentileofscore(color_var, value, kind='rank') for value in color_var]
    norm = plt.Normalize(vmin=0, vmax=100)
    colors = plt.cm.bwr(norm(percentiles))

    plt.figure(figsize=(10, 7))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=2, color=colors)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=plt.gca(), label=color_name)
    plt.title(f'PACMAP Projection colored by {color_name} intensity')
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    logger.info(f"Saved PACMAP plot to: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PACMAP on FCS data and save plot.")
    parser.add_argument(
        "fcs_path", nargs="?", default="20250520_ALL_Patient_BM_sort.fcs",
        help="Path to the .fcs file (default: 20250520_ALL_Patient_BM_sort.fcs)"
    )
    parser.add_argument(
        "output_image_path", nargs="?", default="20250716_an_replication.png",
        help="Path to save the output PNG (default: 20250716_an_replication.png)"
    )
    args = parser.parse_args()

    run_pacmap_on_fcs(args.fcs_path, args.output_image_path)