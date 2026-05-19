"""
Bridge ATTRICI counterfactual climate into DiD-style impact decomposition.

Separates observed yield changes into (a) climate-change-attributable loss and
(b) intervention-attributable avoided loss, using paired Monte Carlo dropout on
:class:`models.yield_surrogate.YieldSurrogateModel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

if TYPE_CHECKING:
    import xarray as xr

    from models.yield_surrogate import YieldSurrogateModel

# DiD / legacy surrogate channel order (see yield_surrogate._LEGACY_4_NAMES)
_DID_CLIMATE_VARS: tuple[str, ...] = ("tmax", "tmin", "precip", "srad")
_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "tmax": ("tmax", "tasmax", "tas"),
    "tmin": ("tmin", "tasmin"),
    "precip": ("precip", "pr"),
    "srad": ("srad", "rsds"),
}
_SEQUENCE_LENGTH = 365


def _resolve_var(ds: xr.Dataset, canonical: str) -> str:
    if canonical in ds.data_vars:
        return canonical
    for alt in _VAR_ALIASES.get(canonical, (canonical,)):
        if alt in ds.data_vars:
            return alt
    raise KeyError(
        f"Climate variable {canonical!r} not in dataset "
        f"(tried {_VAR_ALIASES.get(canonical, (canonical,))}); "
        f"available: {list(ds.data_vars)})"
    )


def _lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.dims or "latitude" in ds.coords:
        return "latitude", "longitude"
    if "lat" in ds.dims or "lat" in ds.coords:
        return "lat", "lon"
    raise ValueError(f"No latitude/longitude coordinates in dataset: {list(ds.coords)}")


def extract_daily_climate_4ch(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    year: int,
) -> np.ndarray:
    """
    Nearest-grid daily climate for one site-year.

    Returns
    -------
    numpy.ndarray
        ``[365, 4]`` in order ``(tmax, tmin, precip, srad)``, float32.
    """
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
    annual = annual.isel(time=slice(0, _SEQUENCE_LENGTH))

    channels: list[np.ndarray] = []
    for canonical in _DID_CLIMATE_VARS:
        var_name = _resolve_var(annual, canonical)
        channels.append(np.asarray(annual[var_name].values, dtype=np.float32))
    return np.stack(channels, axis=-1)


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
    Paired MC dropout: same ``manual_seed`` before each factual/counterfactual forward.

    Returns means, marginal stds, and SE of the mean difference with paired covariance.
    """
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
    Paired seeds align dropout masks between factual and counterfactual forwards.

    Parameters
    ----------
    factual_climate, counterfactual_climate:
        Daily gridded climate (must contain tmax/tmin/precip/srad or aliases).
    yield_model:
        Trained :class:`~models.yield_surrogate.YieldSurrogateModel`.
    static_features:
        ``[N, 10]`` static covariates aligned with ``farm_coords`` rows.
    farm_coords:
        Columns ``farm_id``, ``lat``, ``lon``, ``year``.
    n_mc_samples:
        Monte Carlo dropout passes per site-year.
    device:
        Torch device for inference.

    Returns
    -------
    pandas.DataFrame
        Columns: ``farm_id``, ``year``, ``y_factual_mean``, ``y_factual_std``,
        ``y_cf_mean``, ``y_cf_std``, ``climate_loss_tpha``, ``climate_loss_se``.
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

        factual_4 = extract_daily_climate_4ch(factual_climate, lat, lon, year)
        cf_4 = extract_daily_climate_4ch(counterfactual_climate, lat, lon, year)

        climate_f = torch.from_numpy(factual_4).unsqueeze(0)
        climate_cf = torch.from_numpy(cf_4).unsqueeze(0)
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

    Parameters
    ----------
    did_att:
        DiD average treatment effect on the treated (t/ha).
    did_att_ci:
        ``(lower, upper)`` confidence interval for ``did_att``.
    climate_loss_df:
        Output of :func:`climate_attributable_loss`.
    n_bootstrap:
        Resamples for the climate-attributable CI.
    random_state:
        RNG seed for bootstrap.

    Returns
    -------
    dict
        Keys: ``intervention_att``, ``intervention_att_ci``, ``climate_attributable_mean``,
        ``climate_attributable_ci``, ``total_avoided_loss``.
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

    # Intervention buffers climate stress: report combined avoided loss
    total_avoided = float(did_att + climate_mean)

    return {
        "intervention_att": float(did_att),
        "intervention_att_ci": (float(did_att_ci[0]), float(did_att_ci[1])),
        "climate_attributable_mean": climate_mean,
        "climate_attributable_ci": climate_ci,
        "total_avoided_loss": total_avoided,
    }
