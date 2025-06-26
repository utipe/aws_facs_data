import numpy as np
import faiss

def estimate_density_faiss(data: np.ndarray, k: int = 30) -> np.ndarray:
    """
    Estimate local density using FAISS (Euclidean distance).
    Returns inverse mean distance to k nearest neighbors.
    """
    data = data.astype(np.float32)
    n_samples, dim = data.shape

    # Build FAISS index
    index = faiss.IndexFlatL2(dim)  # L2 = Euclidean distance
    index.add(data)

    # Search for k+1 neighbors (including self)
    dists, _ = index.search(data, k + 1)
    dists = np.sqrt(dists[:, 1:])  # Skip self (distance = 0)

    density = 1.0 / (np.mean(dists, axis=1) + 1e-8)
    return density


def spade_density_downsample_faiss(
    data: np.ndarray,
    k: int = 30,
    exclude_pctile: float = 1.0,
    target_pctile: float = 5.0,
    desired_samples: int = None,
    random_state: int = 42
) -> np.ndarray:
    """
    SPADE-style density-based downsampling using FAISS for kNN.
    """
    rng = np.random.default_rng(seed=random_state)
    data = np.asarray(data, dtype=np.float32)
    n = len(data)

    if n == 0:
        return data

    # Step 1: Density estimation
    density = estimate_density_faiss(data, k=k)

    # Step 2: Compute thresholds
    exclude_thresh = np.percentile(density, exclude_pctile)
    target_thresh = np.percentile(density, target_pctile)

    # Step 3: Mask data
    keep_mask = (density >= exclude_thresh) & (density <= target_thresh)
    sample_mask = density > target_thresh

    keep_all = data[keep_mask]
    sample_pool = data[sample_mask]

    if desired_samples is not None:
        remaining = desired_samples - len(keep_all)
        if remaining <= 0:
            indices = rng.choice(len(keep_all), size=desired_samples, replace=False)
            return keep_all[indices]
        sample_count = min(remaining, len(sample_pool))
    else:
        sample_count = int(len(sample_pool) * (target_pctile / (100 - exclude_pctile)))

    if sample_count > 0:
        sample_indices = rng.choice(len(sample_pool), size=sample_count, replace=False)
        sampled = sample_pool[sample_indices]
        result = np.vstack((keep_all, sampled))
    else:
        result = keep_all

    return result.astype(np.float32, copy=False)
