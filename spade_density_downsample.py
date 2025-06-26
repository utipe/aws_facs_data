import numpy as np
from sklearn.neighbors import NearestNeighbors

def spade_density_downsample(
    data: np.ndarray,
    k: int = 30,
    exclude_pctile: float = 1.0,
    target_pctile: float = 5.0,
    desired_samples: int = None,
    random_state: int = 42
) -> np.ndarray:
    """
    Match SPADE's R-style density-dependent downsampling.

    Parameters:
    - data: ndarray of shape (n_samples, n_features)
    - k: number of neighbors for density estimation
    - exclude_pctile: % of lowest-density cells to exclude (e.g., 1.0)
    - target_pctile: % of lowest-density cells to keep fully (e.g., 5.0)
    - desired_samples: total number of cells to keep (optional)
    - random_state: for reproducibility

    Returns:
    - downsampled_data: ndarray of shape (~desired_samples, n_features)
    """
    np.random.seed(random_state)

    # Estimate density using mean kNN distances
    nn = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nn.kneighbors(data)
    density = 1 / (np.mean(distances, axis=1) + 1e-8)

    # Compute density percentiles
    exclude_thresh = np.percentile(density, exclude_pctile)
    target_thresh = np.percentile(density, target_pctile)

    # Step 1: Exclude lowest-density cells
    mask_exclude = density >= exclude_thresh
    data = data[mask_exclude]
    density = density[mask_exclude]

    # Step 2: Keep low-density cells fully (≤ target_thresh)
    keep_mask = density <= target_thresh
    keep_data = data[keep_mask]
    keep_indices = np.where(keep_mask)[0]

    # Step 3: Random sample from high-density region
    sample_mask = ~keep_mask
    sample_data = data[sample_mask]
    sample_indices = np.where(sample_mask)[0]

    # How many to sample?
    if desired_samples is not None:
        num_to_sample = desired_samples - len(keep_data)
        if num_to_sample < 0:
            # Too many low-density points, randomly reduce
            final_indices = np.random.choice(keep_indices, desired_samples, replace=False)
            return data[final_indices]
    else:
        # If not specified, sample to match % of target vs rest
        total = len(keep_data) + len(sample_data)
        target_fraction = target_pctile / (100 - exclude_pctile)
        num_to_sample = int(target_fraction * total) - len(keep_data)

    num_to_sample = min(num_to_sample, len(sample_data))
    if num_to_sample <= 0:
        return keep_data

    sampled_indices = np.random.choice(len(sample_data), num_to_sample, replace=False)
    downsampled_data = np.vstack([keep_data, sample_data[sampled_indices]])

    return downsampled_data
