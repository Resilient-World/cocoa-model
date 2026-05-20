"""
Bridge ATTRICI counterfactual climate into DiD-style impact decomposition.

Separates observed yield changes into (a) climate-change-attributable loss and
(b) intervention-attributable avoided loss, using paired Monte Carlo dropout on
:class:`models.yield_surrogate.YieldSurrogateModel` (11-channel ERA5 / ATTRICI stack).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from data.era5_ingest import (
    FAO_ALBEDO,
    FAO_GAMMA,
    KELVIN_OFFSET,
    MAGNUS_A,
    MAGNUS_B,
    MAGNUS_C,
    WIND10_TO_WIND2_FACTOR,
)
from models.yield_surrogate import CLIMATE_CHANNEL_NAMES, CLIMATE_IDX, N_CLIMATE_CHANNELS

if TYPE_CHECKING:
    import xarray as xr

    from models.yield_surrogate import YieldSurrogateModel

# Legacy DiD four-channel order (deprecated)
_LEGACY_4_NAMES: tuple[str, ...] = ("tmax", "tmin", "precip", "srad")

_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "tmax": ("tmax", "tasmax", "tas"),
    "tmin": ("tmin", "tasmin"),
    "tmean": ("tmean", "tas"),
    "precip": ("precip", "pr"),
    "srad": ("srad", "rsds"),
    "rh_mean": ("rh_mean", "hurs"),
    "wind10m": ("wind10m", "sfcwind"),
    "vpd": ("vpd", "vpd_mean"),
    "et0": ("et0",),
    "sm_root": ("sm_root",),
    "co2_ppm": ("co2_ppm",),
}

_SEQUENCE_LENGTH = 365


def _saturation_vapor_pressure_kpa(tmean_c: np.ndarray) -> np.ndarray:
    return MAGNUS_A * np.exp(MAGNUS_B * tmean_c / (MAGNUS_C + tmean_c))


def _vpd_kpa(tmean_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    rh = np.clip(rh_pct, 0.0, 100.0)
    es = _saturation_vapor_pressure_kpa(tmean_c)
    return es * (1.0 - rh / 100.0)


def _fao_et0_numpy(
    tmean_c: np.ndarray,
    rh_pct: np.ndarray,
    wind10m: np.ndarray,
    srad_mj: np.ndarray,
) -> np.ndarray:
    """FAO-56 Penman–Monteith reference ET0 (mm/day); mirrors :func:`data.era5_ingest._fao_et0_daily`."""
    es = _saturation_vapor_pressure_kpa(tmean_c)
    ea = es * (rh_pct / 100.0)
    vpd = np.maximum(es - ea, 0.0)
    delta = es * MAGNUS_B * MAGNUS_C / (tmean_c + MAGNUS_C) ** 2
    u2 = wind10m * WIND10_TO_WIND2_FACTOR
    rn = srad_mj * (1.0 - FAO_ALBEDO)
    t_k = tmean_c + KELVIN_OFFSET
    num_rad = delta * rn * 0.408
    num_aero = FAO_GAMMA * (900.0 / t_k) * u2 * vpd
    den = delta + FAO_GAMMA * (1.0 + 0.34 * u2)
    return (num_rad + num_aero) / den


def _resolve_var(ds: xr.Dataset, canonical: str) -> str:
    cf_name = f"{canonical}_cf"
    candidates = (cf_name, canonical) + _VAR_ALIASES.get(canonical, (canonical,))
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        if name in ds.data_vars:
            return name
    raise KeyError(
        f"Climate variable {canonical!r} not in dataset "
        f"(tried {candidates}); available: {list(ds.data_vars)}"
    )


def _lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.dims or "latitude" in ds.coords:
        return "latitude", "longitude"
    if "lat" in ds.dims or "lat" in ds.coords:
        return "lat", "lon"
    raise ValueError(f"No latitude/longitude coordinates in dataset: {list(ds.coords)}")


def _select_site_year(ds: xr.Dataset, lat: float, lon: float, year: int) -> xr.Dataset:
    lat_name, lon_name = _lat_lon_names(ds)
    point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    if "time" not in point.dims and "time" not in point.coords:
        raise ValueError("Dataset has no time dimension")
    annual = point.sel(time=point["time"].dt.year == year)
    n_days = int(annual.sizes.get("time", 0))
    if n_days < _SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {_SEQUENCE_LENGTH} daily timesteps for year={year}, got {n_days}"
        )
    return annual.isel(time=slice(0, _SEQUENCE_LENGTH))


def _daily_values(annual: xr.Dataset, canonical: str) -> np.ndarray:
    var_name = _resolve_var(annual, canonical)
    return np.asarray(annual[var_name].values, dtype=np.float32).reshape(-1)


def extract_daily_climate_11ch(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    year: int,
    *,
    factual_reference: xr.Dataset | None = None,
) -> np.ndarray:
    """
    Nearest-grid daily climate for one site-year in :data:`CLIMATE_CHANNEL_NAMES` order.

    Detrended state variables (``tmax``, ``tmin``, ``precip``, ``srad``, ``rh_mean``,
    ``wind10m``, ``vpd`` when present) are read from ``ds`` with ISIMIP/ERA5 aliases
    (``tasmax``, ``hurs``, ``rsds``, ``sfcwind``, ``pr``, and ``*_cf`` counterfactual names).

    Variables ATTRICI does not detrend are handled as follows:

    - ``tmean``: always ``(tmax + tmin) / 2`` from the daily series in ``ds``.
    - ``et0``: always recomputed with FAO-56 Penman–Monteith using ``tmean``, ``rh_mean``,
      ``wind10m``, and ``srad`` from ``ds`` (same formulation as :mod:`data.era5_ingest`).
    - ``sm_root`` and ``co2_ppm``: taken from ``factual_reference`` when provided
      (counterfactual runs); otherwise from ``ds``. These drivers are assumed unaffected
      by the GMT detrending experiment.

    Returns
    -------
    numpy.ndarray
        ``[365, 11]`` float32, channel order matching :class:`~models.yield_surrogate.YieldSurrogateModel`.
    """
    annual = _select_site_year(ds, lat, lon, year)

    tmax = _daily_values(annual, "tmax")
    tmin = _daily_values(annual, "tmin")
    precip = np.maximum(_daily_values(annual, "precip"), 0.0)
    srad = np.maximum(_daily_values(annual, "srad"), 0.0)
    rh = np.clip(_daily_values(annual, "rh_mean"), 0.0, 100.0)
    wind = np.maximum(_daily_values(annual, "wind10m"), 0.0)

    tmean = 0.5 * (tmax + tmin)
    vpd = _vpd_kpa(tmean, rh)
    et0 = np.maximum(_fao_et0_numpy(tmean, rh, wind, srad), 0.0)

    if factual_reference is not None:
        ref = _select_site_year(factual_reference, lat, lon, year)
        sm_root = _daily_values(ref, "sm_root")
        co2_ppm = _daily_values(ref, "co2_ppm")
    else:
        sm_root = _daily_values(annual, "sm_root")
        co2_ppm = _daily_values(annual, "co2_ppm")

    channel_map = {
        "tmax": tmax,
        "tmin": tmin,
        "tmean": tmean,
        "precip": precip,
        "srad": srad,
        "vpd": vpd,
        "et0": et0,
        "sm_root": sm_root,
        "wind10m": wind,
        "rh_mean": rh,
        "co2_ppm": co2_ppm,
    }
    return np.stack([channel_map[name] for name in CLIMATE_CHANNEL_NAMES], axis=-1).astype(
        np.float32
    )


def extract_daily_climate_4ch(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    year: int,
) -> np.ndarray:
    """
    Deprecated: use :func:`extract_daily_climate_11ch`.

    Returns ``[365, 4]`` in legacy order ``(tmax, tmin, precip, srad)`` with other
    channels zero-filled (not suitable for :class:`~models.yield_surrogate.YieldSurrogateModel`).
    """
    warnings.warn(
        "extract_daily_climate_4ch is deprecated; use extract_daily_climate_11ch for the "
        "full ERA5 11-channel stack required by YieldSurrogateModel.",
        DeprecationWarning,
        stacklevel=2,
    )
    full = extract_daily_climate_11ch(ds, lat, lon, year)
    out = np.zeros((_SEQUENCE_LENGTH, N_CLIMATE_CHANNELS), dtype=np.float32)
    for i, name in enumerate(_LEGACY_4_NAMES):
        out[:, CLIMATE_IDX[name]] = full[:, CLIMATE_IDX[name]]
    return out[:, [CLIMATE_IDX[n] for n in _LEGACY_4_NAMES]]


@torch.no_grad()
def _paired_mc_yields(
    yield_model: YieldSurrogateModel,
    climate_factual: Tensor,
    climate_counterfactual: Tensor,
    static: Tensor,
    *,
    n_mc_samples: int,
    seed: int,
    device: str,
) -> tuple[float, float, float, float, float]:
    """
    Paired MC dropout on factual vs counterfactual climate tensors.

    Parameters
    ----------
    climate_factual, climate_counterfactual:
        ``[B, 365, 11]`` daily stacks (see :data:`CLIMATE_CHANNEL_NAMES`).
    """
    if climate_factual.shape[-1] != N_CLIMATE_CHANNELS:
        raise ValueError(
            f"climate_factual last dim must be {N_CLIMATE_CHANNELS}, got {climate_factual.shape[-1]}"
        )
    if climate_counterfactual.shape[-1] != N_CLIMATE_CHANNELS:
        raise ValueError(
            f"climate_counterfactual last dim must be {N_CLIMATE_CHANNELS}, "
            f"got {climate_counterfactual.shape[-1]}"
        )

    yield_model.eval()
    yield_model.to(device)
    climate_factual = climate_factual.to(device)
    climate_counterfactual = climate_counterfactual.to(device)
    static = static.to(device)

    samples_f: list[Tensor] = []
    samples_cf: list[Tensor] = []

    for i in range(n_mc_samples):
        torch.manual_seed(seed + i)
        samples_f.append(yield_model(climate_factual, static).squeeze(-1))
        torch.manual_seed(seed + i)
        samples_cf.append(yield_model(climate_counterfactual, static).squeeze(-1))

    stack_f = torch.stack(samples_f, dim=0)
    stack_cf = torch.stack(samples_cf, dim=0)

    y_factual_mean = float(stack_f.mean().item())
    y_cf_mean = float(stack_cf.mean().item())

    if n_mc_samples > 1:
        y_factual_std = float(stack_f.std(unbiased=False).item())
        y_cf_std = float(stack_cf.std(unbiased=False).item())
        cov = float(
            ((stack_f - stack_f.mean()) * (stack_cf - stack_cf.mean())).mean().item()
        )
        climate_loss_se = float(
            np.sqrt(max(0.0, y_factual_std**2 + y_cf_std**2 - 2.0 * cov))
        )
    else:
        y_factual_std = 0.0
        y_cf_std = 0.0
        climate_loss_se = 0.0

    return y_factual_mean, y_factual_std, y_cf_mean, y_cf_std, climate_loss_se


def _farm_seed(farm_id: Any, year: int) -> int:
    return int(hash((str(farm_id), int(year))) % (2**31))


def climate_attributable_loss(
    factual_climate: xr.Dataset,
    counterfactual_climate: xr.Dataset,
    yield_model: YieldSurrogateModel,
    static_features: np.ndarray,
    farm_coords: pd.DataFrame,
    n_mc_samples: int = 50,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Per-farm-year counterfactual yield gap with paired MC-dropout uncertainty.

    Gap (tonnes/ha) is ``E[Y | factual climate] - E[Y | counterfactual climate]``.
    Climate tensors are built with :func:`extract_daily_climate_11ch` (11 channels).

    Parameters
    ----------
    factual_climate, counterfactual_climate:
        Daily gridded ERA5 / ATTRICI datasets (detrended ``*_cf`` variables on the
        counterfactual side).
    """
    required = {"farm_id", "lat", "lon", "year"}
    missing = required - set(farm_coords.columns)
    if missing:
        raise ValueError(f"farm_coords missing columns: {sorted(missing)}")

    n_rows = len(farm_coords)
    if static_features.shape != (n_rows, 10):
        raise ValueError(
            f"static_features must be shape ({n_rows}, 10), got {static_features.shape}"
        )

    records: list[dict[str, Any]] = []
    for row_idx in range(n_rows):
        row = farm_coords.iloc[row_idx]
        farm_id = row["farm_id"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        year = int(row["year"])
        static_row = static_features[row_idx]

        factual_11 = extract_daily_climate_11ch(factual_climate, lat, lon, year)
        cf_11 = extract_daily_climate_11ch(
            counterfactual_climate,
            lat,
            lon,
            year,
            factual_reference=factual_climate,
        )

        climate_f = torch.from_numpy(factual_11).unsqueeze(0)
        climate_cf = torch.from_numpy(cf_11).unsqueeze(0)
        static_t = torch.from_numpy(static_row.astype(np.float32)).unsqueeze(0)

        seed = _farm_seed(farm_id, year)
        y_f_mean, y_f_std, y_cf_mean, y_cf_std, loss_se = _paired_mc_yields(
            yield_model,
            climate_f,
            climate_cf,
            static_t,
            n_mc_samples=n_mc_samples,
            seed=seed,
            device=device,
        )

        climate_loss = y_f_mean - y_cf_mean
        records.append(
            {
                "farm_id": farm_id,
                "year": year,
                "y_factual_mean": y_f_mean,
                "y_factual_std": y_f_std,
                "y_cf_mean": y_cf_mean,
                "y_cf_std": y_cf_std,
                "climate_loss_tpha": climate_loss,
                "climate_loss_se": loss_se,
            }
        )

    return pd.DataFrame(records)


def decompose_avoided_loss(
    did_att: float,
    did_att_ci: tuple[float, float],
    climate_loss_df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, float | tuple[float, float]]:
    """
    Decompose total avoided loss into intervention (DiD) and climate-attributable parts.

    ``intervention_att`` is the DiD ATT (unchanged). ``climate_attributable_mean`` is
    the mean per-farm-year climate yield gap. ``total_avoided_loss`` sums the
    intervention effect with the climate component when interventions buffer
    climate stress (additive decomposition for reporting).
    """
    if climate_loss_df.empty:
        raise ValueError("climate_loss_df is empty")

    if "climate_loss_tpha" not in climate_loss_df.columns:
        raise ValueError("climate_loss_df must contain column 'climate_loss_tpha'")

    losses = climate_loss_df["climate_loss_tpha"].to_numpy(dtype=float)
    climate_mean = float(np.mean(losses))

    rng = np.random.default_rng(random_state)
    n = len(losses)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(losses[idx].mean())

    climate_ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))

    total_avoided = float(did_att + climate_mean)

    return {
        "intervention_att": float(did_att),
        "intervention_att_ci": (float(did_att_ci[0]), float(did_att_ci[1])),
        "climate_attributable_mean": climate_mean,
        "climate_attributable_ci": climate_ci,
        "total_avoided_loss": total_avoided,
    }
