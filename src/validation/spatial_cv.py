"""
Spatial cross-validation (Roberts et al. 2017, Ecography, doi:10.1111/ecog.02881).

Block CV, buffered leave-one-out, and residual variogram helpers for choosing block size.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Literal

import numpy as np
from pyproj import Transformer

Strategy = Literal["checkerboard", "random_assignment", "optimised_random"]

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) from one point to arrays."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def project_coords_km(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 to UTM (per-point zone) returning easting/northing in km."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    east_km = np.empty(len(lats))
    north_km = np.empty(len(lats))
    for i, (la, lo) in enumerate(zip(lats, lons, strict=True)):
        epsg = _utm_epsg(float(la), float(lo))
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        e, n = transformer.transform(float(lo), float(la))
        east_km[i] = e / 1000.0
        north_km[i] = n / 1000.0
    return east_km, north_km


def block_ids_from_coords(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    block_size_km: float,
) -> np.ndarray:
    """Integer block id per sample from UTM-km grid."""
    east_km, north_km = project_coords_km(lats, lons)
    row = np.floor(north_km / block_size_km).astype(np.int64)
    col = np.floor(east_km / block_size_km).astype(np.int64)
    return row * 1_000_003 + col


def recommend_block_size_km(range_km: float, *, floor_km: float = 25.0) -> float:
    """Roberts Step 1: block size >= 1.5 × variogram range."""
    return float(max(range_km * 1.5, floor_km))


def compute_residual_variogram(
    predictions: np.ndarray,
    residuals: np.ndarray,
    coords: np.ndarray,
    *,
    n_lags: int = 15,
) -> dict[str, float]:
    """
    Empirical variogram of residuals via skgstat.

    Parameters
    ----------
    coords:
        ``[N, 2]`` with columns ``(easting_m, northing_m)`` or WGS84 ``(lon, lat)``.
    """
    import skgstat as skg

    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    res = np.asarray(residuals, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] == 2 and np.all(np.abs(coords[:, 0]) <= 180):
        lons, lats = coords[:, 0], coords[:, 1]
        east_km, north_km = project_coords_km(lats, lons)
        coords_m = np.column_stack([east_km * 1000.0, north_km * 1000.0])
    else:
        coords_m = coords

    if len(res) < 10:
        return {"range_km": 50.0, "sill": float(np.var(res)), "nugget": 0.0}

    v = skg.Variogram(
        coords_m,
        res,
        n_lags=min(n_lags, max(5, len(res) // 10)),
        maxlag="median",
        model="spherical",
    )
    fit = v.fit()
    range_km = float(getattr(fit, "len", getattr(fit, "range", 50.0))) / 1000.0
    return {
        "range_km": range_km,
        "sill": float(getattr(fit, "sill", np.var(res))),
        "nugget": float(getattr(fit, "nugget", 0.0)),
    }


def spatial_holdout_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    fraction: float = 0.10,
    seed: int = 42,
    block_size_km: float = 50.0,
) -> np.ndarray:
    """Backward-compatible ~fraction holdout using one fold of block CV."""
    splitter = SpatialBlockSplit(
        block_size_km=block_size_km,
        n_folds=max(5, int(1.0 / max(fraction, 0.05))),
        strategy="random_assignment",
        seed=seed,
    )
    folds = list(splitter.split(lats, lons))
    if not folds:
        return np.zeros(len(lats), dtype=bool)
    _, test_idx = folds[0]
    mask = np.zeros(len(lats), dtype=bool)
    mask[test_idx] = True
    return mask


def _assign_blocks_to_folds(
    block_ids: np.ndarray,
    n_folds: int,
    strategy: Strategy,
    *,
    seed: int,
    residuals: np.ndarray | None = None,
    n_candidates: int = 200,
) -> np.ndarray:
    """Return fold id per sample (0 .. n_folds-1)."""
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)
    rng = np.random.default_rng(seed)

    if strategy == "checkerboard":
        fold_map: dict[int, int] = {}
        for bid in unique_blocks:
            row = bid // 1_000_003
            col = bid % 1_000_003
            fold_map[int(bid)] = int((row + col) % n_folds)
        return np.array([fold_map[int(b)] for b in block_ids], dtype=np.int64)

    if strategy == "random_assignment":
        block_folds = rng.integers(0, n_folds, size=n_blocks)
        fold_by_block = {int(u): int(f) for u, f in zip(unique_blocks, block_folds, strict=True)}
        return np.array([fold_by_block[int(b)] for b in block_ids], dtype=np.int64)

    # optimised_random
    best_score = float("inf")
    best_assignment = rng.integers(0, n_folds, size=n_blocks)
    res = np.zeros(len(block_ids)) if residuals is None else np.asarray(residuals, dtype=np.float64)
    for _ in range(n_candidates):
        block_folds = rng.integers(0, n_folds, size=n_blocks)
        fold_by_block = {int(u): int(f) for u, f in zip(unique_blocks, block_folds, strict=True)}
        sample_folds = np.array([fold_by_block[int(b)] for b in block_ids])
        score = _morans_i_proxy(res, sample_folds, n_folds)
        if score < best_score:
            best_score = score
            best_assignment = block_folds
    fold_by_block = {int(u): int(f) for u, f in zip(unique_blocks, best_assignment, strict=True)}
    return np.array([fold_by_block[int(b)] for b in block_ids], dtype=np.int64)


def _morans_i_proxy(residuals: np.ndarray, fold_ids: np.ndarray, n_folds: int) -> float:
    """Lower is better: variance of fold means (proxy for spatial leakage)."""
    means = []
    for f in range(n_folds):
        m = fold_ids == f
        if m.any():
            means.append(float(residuals[m].mean()))
    return float(np.var(means)) if means else 0.0


class SpatialBlockSplit:
    """K-fold spatial block CV (Roberts et al. 2017)."""

    def __init__(
        self,
        *,
        block_size_km: float = 50.0,
        buffer_km: float = 0.0,
        n_folds: int = 5,
        strategy: Strategy = "checkerboard",
        seed: int = 42,
        n_candidates: int = 200,
    ) -> None:
        self.block_size_km = block_size_km
        self.buffer_km = buffer_km
        self.n_folds = max(2, n_folds)
        self.strategy = strategy
        self.seed = seed
        self.n_candidates = n_candidates

    def split(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        *,
        residuals: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        lats = np.asarray(lats, dtype=np.float64)
        lons = np.asarray(lons, dtype=np.float64)
        n = len(lats)
        block_ids = block_ids_from_coords(lats, lons, block_size_km=self.block_size_km)
        fold_ids = _assign_blocks_to_folds(
            block_ids,
            self.n_folds,
            self.strategy,
            seed=self.seed,
            residuals=residuals,
            n_candidates=self.n_candidates,
        )
        indices = np.arange(n)
        for fold in range(self.n_folds):
            test_mask = fold_ids == fold
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]
            if self.buffer_km > 0 and len(test_idx) > 0:
                keep = np.ones(len(train_idx), dtype=bool)
                for ti in test_idx:
                    d = haversine_km(lats[ti], lons[ti], lats[train_idx], lons[train_idx])
                    keep &= d > self.buffer_km
                train_idx = train_idx[keep]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_folds


class BufferedLOO:
    """Leave-one-out with spatial buffer (Le Rest et al. 2014)."""

    def __init__(self, *, buffer_km: float = 50.0) -> None:
        self.buffer_km = buffer_km

    def split(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        *,
        residuals: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        del residuals
        lats = np.asarray(lats, dtype=np.float64)
        lons = np.asarray(lons, dtype=np.float64)
        n = len(lats)
        indices = np.arange(n)
        for i in range(n):
            d = haversine_km(lats[i], lons[i], lats, lons)
            train_mask = (d > self.buffer_km) & (indices != i)
            train_idx = indices[train_mask]
            test_idx = indices[i : i + 1]
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, n_samples: int | None = None) -> int:
        return int(n_samples) if n_samples is not None else 0
