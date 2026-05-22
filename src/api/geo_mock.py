"""Mock geospatial retrieval of climate and soil features for a farm location."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor

from models.yield_surrogate import N_STATIC_SITE, pack_tree_age_static

SEQUENCE_LENGTH = 365
CLIMATE_FEATURES = 11
STATIC_FEATURES = N_STATIC_SITE

# Aligns with models.yield_surrogate.CLIMATE_CHANNEL_NAMES
_CLIMATE_TMAX = 0
_CLIMATE_TMIN = 1
_CLIMATE_TMEAN = 2
_CLIMATE_PRECIP = 3
_CLIMATE_SRAD = 4
_CLIMATE_VPD = 5
_CLIMATE_ET0 = 6
_CLIMATE_SM_ROOT = 7
_CLIMATE_WIND10M = 8
_CLIMATE_RH = 9
_CLIMATE_CO2 = 10

MAGNUS_A = 0.61094
MAGNUS_B = 17.625
MAGNUS_C = 243.04
DEFAULT_AWC_MM = 150.0


def _location_seed(lat: float, lon: float) -> int:
    payload = f"{lat:.6f},{lon:.6f}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def _saturation_vapor_pressure_kpa(t_c: np.ndarray) -> np.ndarray:
    return MAGNUS_A * np.exp(MAGNUS_B * t_c / (MAGNUS_C + t_c))


def fetch_climate_and_soil(lat: float, lon: float) -> tuple[Tensor, Tensor]:
    """
    Deterministic mock of ERA5-Land–style daily climate and static site features.

    Returns
    -------
    climate:
        ``[1, 365, 11]`` — physical units, channel order:
        tmax, tmin, tmean, precip (mm/d), srad (MJ/m²/d), vpd (kPa), et0 (mm/d),
        sm_root (m³/m³), wind10m (m/s), rh_mean (%), co2_ppm.
    static:
        ``[1, 13]`` — index 0 = AWC (mm); indices 10–12 = tree-age cohort features.
    """
    rng = np.random.default_rng(_location_seed(lat, lon))

    day_of_year = np.arange(SEQUENCE_LENGTH, dtype=np.float32)
    seasonal = np.sin(2 * np.pi * day_of_year / 365.0)

    t_max = 30.0 + 3.0 * seasonal + rng.normal(0, 0.5, SEQUENCE_LENGTH)
    t_min = t_max - rng.uniform(6.0, 10.0, SEQUENCE_LENGTH)
    t_mean = 0.5 * (t_max + t_min)
    precip = np.clip(rng.gamma(2.0, 4.0, SEQUENCE_LENGTH), 0.0, 80.0)
    srad = np.clip(12.0 + 4.0 * seasonal + rng.normal(0, 0.3, SEQUENCE_LENGTH), 5.0, 25.0)

    rh_mean = np.clip(78.0 + 8.0 * seasonal + rng.normal(0, 3.0, SEQUENCE_LENGTH), 55.0, 98.0)
    es = _saturation_vapor_pressure_kpa(t_mean)
    ea = es * rh_mean / 100.0
    vpd = np.clip(es - ea, 0.05, 3.5)

    et0 = np.clip(3.2 + 0.4 * seasonal + rng.normal(0, 0.15, SEQUENCE_LENGTH), 1.5, 6.0)
    sm_root = np.clip(
        0.28 + 0.04 * np.cos(seasonal) + rng.normal(0, 0.02, SEQUENCE_LENGTH), 0.12, 0.42
    )
    wind10m = np.clip(2.0 + rng.normal(0, 0.35, SEQUENCE_LENGTH), 0.5, 6.0)

    co2_base = 415.0 + (lat * 0.1) + (lon * 0.05)
    co2_ppm = co2_base + rng.normal(0, 2.0, SEQUENCE_LENGTH)

    climate = np.zeros((SEQUENCE_LENGTH, CLIMATE_FEATURES), dtype=np.float32)
    climate[:, _CLIMATE_TMAX] = t_max.astype(np.float32)
    climate[:, _CLIMATE_TMIN] = t_min.astype(np.float32)
    climate[:, _CLIMATE_TMEAN] = t_mean.astype(np.float32)
    climate[:, _CLIMATE_PRECIP] = precip.astype(np.float32)
    climate[:, _CLIMATE_SRAD] = srad.astype(np.float32)
    climate[:, _CLIMATE_VPD] = vpd.astype(np.float32)
    climate[:, _CLIMATE_ET0] = et0.astype(np.float32)
    climate[:, _CLIMATE_SM_ROOT] = sm_root.astype(np.float32)
    climate[:, _CLIMATE_WIND10M] = wind10m.astype(np.float32)
    climate[:, _CLIMATE_RH] = rh_mean.astype(np.float32)
    climate[:, _CLIMATE_CO2] = co2_ppm.astype(np.float32)

    static = rng.uniform(0.0, 1.0, STATIC_FEATURES).astype(np.float32)
    static[0] = DEFAULT_AWC_MM
    static[1] = np.clip((lat + 10.0) / 50.0, 0.0, 1.0)
    static[2] = np.clip((lon + 20.0) / 60.0, 0.0, 1.0)
    age_norm, cohort, dens_norm = pack_tree_age_static(12.0)
    static[10] = age_norm
    static[11] = cohort
    static[12] = dens_norm

    return (
        torch.from_numpy(climate).unsqueeze(0),
        torch.from_numpy(static).unsqueeze(0),
    )
