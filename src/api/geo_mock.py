"""Mock geospatial retrieval of climate and soil features for a farm location."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor

SEQUENCE_LENGTH = 365
CLIMATE_FEATURES = 4
STATIC_FEATURES = 10


def _location_seed(lat: float, lon: float) -> int:
    payload = f"{lat:.6f},{lon:.6f}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def fetch_climate_and_soil(lat: float, lon: float) -> tuple[Tensor, Tensor]:
    """
    Deterministic mock of ERA5-like climate and static soil/site features.

    Returns
    -------
    climate:
        ``[1, 365, 4]`` — daily max temp, min temp, precip (mm), radiation (MJ/m²).
    static:
        ``[1, 10]`` — normalized site covariates.
    """
    rng = np.random.default_rng(_location_seed(lat, lon))

    # Daily climate: plausible tropical cocoa belt ranges
    day_of_year = np.arange(SEQUENCE_LENGTH, dtype=np.float32)
    seasonal = np.sin(2 * np.pi * day_of_year / 365.0)
    t_max = 30.0 + 3.0 * seasonal + rng.normal(0, 0.5, SEQUENCE_LENGTH)
    t_min = t_max - rng.uniform(6.0, 10.0, SEQUENCE_LENGTH)
    precip = np.clip(rng.gamma(2.0, 4.0, SEQUENCE_LENGTH), 0.0, 80.0)
    radiation = np.clip(12.0 + 4.0 * seasonal + rng.normal(0, 0.3, SEQUENCE_LENGTH), 5.0, 25.0)

    climate = np.stack([t_max, t_min, precip, radiation], axis=-1).astype(np.float32)
    climate = (climate - climate.mean(axis=0)) / (climate.std(axis=0) + 1e-6)

    static = rng.uniform(0.0, 1.0, STATIC_FEATURES).astype(np.float32)
    static[0] = np.clip((lat + 10.0) / 50.0, 0.0, 1.0)
    static[1] = np.clip((lon + 20.0) / 60.0, 0.0, 1.0)

    return (
        torch.from_numpy(climate).unsqueeze(0),
        torch.from_numpy(static).unsqueeze(0),
    )
