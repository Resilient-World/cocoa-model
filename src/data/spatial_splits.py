"""Backward-compatible re-exports; prefer :mod:`validation.spatial_cv`."""

from __future__ import annotations

from validation.spatial_cv import (
    BufferedLOO,
    SpatialBlockSplit,
    compute_residual_variogram,
    recommend_block_size_km,
    spatial_holdout_mask,
)

__all__ = [
    "BufferedLOO",
    "SpatialBlockSplit",
    "compute_residual_variogram",
    "recommend_block_size_km",
    "spatial_holdout_mask",
]
