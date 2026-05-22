"""Spatial block holdout splits for HPO and benchmarks."""

from __future__ import annotations

import numpy as np


def spatial_holdout_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    fraction: float = 0.10,
    seed: int = 42,
) -> np.ndarray:
    """Deterministic spatial block holdout (~fraction of 0.5° cells)."""
    cell_ids = (np.floor(lats * 2.0).astype(np.int64) * 10_000) + np.floor(lons * 2.0).astype(
        np.int64
    )
    unique = np.unique(cell_ids)
    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(unique) * fraction))
    test_cells = set(rng.choice(unique, size=n_test, replace=False).tolist())
    return np.isin(cell_ids, list(test_cells))
