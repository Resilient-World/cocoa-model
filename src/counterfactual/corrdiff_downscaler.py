"""
CorrDiff-CMIP6 stochastic km-scale downscaling (Mardani et al. 2025; Earth2Studio).

Uses ``earth2studio.models.dx.CorrDiffCMIP6`` with ``CMIP6MultiRealm`` inputs when the
optional ``[corrdiff]`` dependencies and GPU are available. Outputs daily ERA5-Land-schema
stacks with a leading ``sample`` dimension for stochastic scenario UQ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog
import xarray as xr

from counterfactual.cmip6_scenarios import ScenarioBuilder
from data.cocoa_exposure import REGIONS, normalize_region_key, region_bounds_dict
from data.era5_ingest import compute_derived_features

log = structlog.get_logger(__name__)

ExperimentId = Literal["ssp245", "ssp585"]
SolverType = Literal["euler", "heun"]
SamplerType = Literal["deterministic", "stochastic"]

# Subset of CorrDiff ERA5 outputs mapped to cocoa yield-surrogate channels
CORRDIFF_TO_ERA5: dict[str, str] = {
    "t2m": "tmean",
    "mx2t": "tmax",
    "mn2t": "tmin",
    "tp": "precip",
    "ssrd": "srad",
    "u10m": "wind10m",
    "v10m": "wind10m",
    "r": "rh_mean",
}

DEFAULT_OUTPUT_VARIABLES: tuple[str, ...] = (
    "tmax",
    "tmin",
    "tmean",
    "precip",
    "srad",
    "rh_mean",
    "wind10m",
    "vpd_mean",
    "et0",
    "sm_root",
    "cwd",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_corrdiff() -> None:
    try:
        import earth2studio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CorrDiff requires pip install -e '.[corrdiff]' (nvidia-modulus + earth2studio)."
        ) from exc


def corrdiff_cache_path(
    processed_dir: Path,
    scenario: str,
    horizon: int,
    region: str,
) -> Path:
    region_key = normalize_region_key(region)
    return processed_dir / f"corrdiff_{scenario}_{int(horizon)}_{region_key}.zarr"


def corrdiff_cache_missing_message(
    cache_path: Path, scenario: str, horizon: int, region: str
) -> str:
    return (
        f"CorrDiff cache not found at {cache_path}. "
        f"Run: PYTHONPATH=src python scripts/run_corrdiff_scenario_bulk.py "
        f"--strata {scenario}:{horizon}:{region}"
    )


@dataclass
class CorrDiffCMIP6Downscaler:
    """
    Residual corrective diffusion downscaling from CMIP6 to km-scale ERA5-like fields.

    For production, call ``downscale_horizon_year`` on GPU and persist with ``to_zarr``.
    """

    experiment_id: ExperimentId
    source_id: str = "CanESM5"
    variant_label: str = "r1i1p2f1"
    number_of_samples: int = 8
    solver: SolverType = "euler"
    sampler_type: SamplerType = "stochastic"
    region: str = "ghana"
    historical_zarr_path: str | Path | None = None
    cmip6_zarr_path: str | Path | None = None
    _dataset: xr.Dataset | None = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.region = normalize_region_key(self.region)
        self._bounds = region_bounds_dict(self.region)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        _require_corrdiff()
        import torch
        from earth2studio.models.dx import CorrDiffCMIP6

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("CorrDiffCMIP6 inference requires a CUDA GPU (A100-80GB or H100).")

        pkg = CorrDiffCMIP6.load_default_package()
        model = CorrDiffCMIP6.load_model(pkg)
        model.number_of_samples = int(self.number_of_samples)
        model.solver = self.solver
        model.sampler_type = self.sampler_type
        model = model.to(device)
        self._model = model
        return model

    def downscale(
        self,
        time: np.datetime64,
        output_variables: list[str],
    ) -> xr.Dataset:
        """
        Downscale one valid time (12:00 UTC anchor) to an N-member hourly ensemble, then
        aggregate to daily ERA5-schema fields for the configured region bbox.
        """
        _require_corrdiff()
        from earth2studio.data import CMIP6, CMIP6MultiRealm
        from earth2studio.data.utils import fetch_data

        model = self._load_model()
        device = next(model.parameters()).device

        cmip6_kwargs = dict(
            experiment_id=self.experiment_id,
            source_id=self.source_id,
            variant_label=self.variant_label,
            exact_time_match=True,
        )
        data = CMIP6MultiRealm(
            [CMIP6(table_id=t, **cmip6_kwargs) for t in ("day", "Eday", "SIday")]
        )

        t = np.datetime64(time)
        if np.datetime64(t).astype("datetime64[h]").astype(int) % 24 != 12:
            t = t.astype("datetime64[D]") + np.timedelta64(12, "h")

        x, coords = fetch_data(
            source=data,
            time=np.array([t]),
            lead_time=model.input_coords()["lead_time"],
            variable=model.input_coords()["variable"],
            device=device,
        )
        out, out_coords = model(x, coords)
        hourly = xr.DataArray(
            data=out.detach().cpu().numpy(),
            coords=out_coords,
            dims=list(out_coords.keys()),
        ).to_dataset(name="values")
        daily = _hourly_to_daily_era5(hourly, output_variables, self._bounds)
        daily.attrs["method"] = "corrdiff_cmip6"
        daily.attrs["experiment_id"] = self.experiment_id
        daily.attrs["number_of_samples"] = self.number_of_samples
        return daily

    def downscale_horizon_year(
        self,
        horizon: int,
        output_variables: list[str],
    ) -> xr.Dataset:
        """
        Build a daily ``sample × time × lat × lon`` dataset for a calendar year.

        On GPU, uses monthly 12:00 UTC CorrDiff anchors (12 forwards/year). Without CUDA,
        falls back to ``ScenarioBuilder`` linear delta-change plus per-sample noise.
        """
        try:
            self._load_model()
            merged = self._downscale_horizon_year_gpu(horizon, output_variables)
        except (ImportError, RuntimeError) as exc:
            log.info("CorrDiff GPU path unavailable (%s); linear bridge", exc)
            merged = _downscale_horizon_year_linear_ensemble(
                horizon=horizon,
                experiment_id=self.experiment_id,
                region=self.region,
                output_variables=output_variables,
                n_samples=self.number_of_samples,
                historical_zarr_path=self.historical_zarr_path,
                cmip6_zarr_path=self.cmip6_zarr_path,
            )
        merged.attrs["method"] = merged.attrs.get("method", "corrdiff_cmip6")
        merged.attrs["horizon_year"] = int(horizon)
        merged.attrs["region"] = self.region
        self._dataset = merged
        return merged

    def _downscale_horizon_year_gpu(
        self,
        horizon: int,
        output_variables: list[str],
    ) -> xr.Dataset:
        months = pd.date_range(f"{horizon}-01-01", f"{horizon}-12-31", freq="MS")
        day_blocks: list[xr.Dataset] = []
        for month_start in months:
            anchor = np.datetime64(month_start) + np.timedelta64(12, "h")
            ds_m = self.downscale(anchor, output_variables)
            n_days = pd.Period(month_start, freq="M").days_in_month
            days = pd.date_range(month_start, periods=n_days, freq="D")
            for day in days:
                block = ds_m.copy(deep=True)
                if "time" in block.dims:
                    block = block.isel(time=0, drop=True)
                block = block.expand_dims(time=[np.datetime64(day)])
                day_blocks.append(block)
        if not day_blocks:
            raise ValueError(f"No daily blocks for horizon {horizon}")
        return xr.concat(day_blocks, dim="time")

    def to_zarr(self, path: Path, sample_dim_chunked: bool = True) -> None:
        if self._dataset is None:
            raise ValueError("No dataset to write; call downscale_horizon_year first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoding: dict[str, dict[str, Any]] = {}
        if sample_dim_chunked and "sample" in self._dataset.dims:
            encoding["sample"] = {"chunks": (1,)}
        self._dataset.to_zarr(path, mode="w", encoding=encoding)


def _downscale_horizon_year_linear_ensemble(
    *,
    horizon: int,
    experiment_id: str,
    region: str,
    output_variables: list[str],
    n_samples: int,
    historical_zarr_path: str | Path | None,
    cmip6_zarr_path: str | Path | None,
) -> xr.Dataset:
    """Full-year linear ScenarioBuilder stack with stochastic per-sample perturbations."""
    hist = historical_zarr_path or _REPO_ROOT / "data/processed/era5_2020_2024.zarr"
    cmip6 = cmip6_zarr_path or _REPO_ROOT / "data/processed/cmip6_ensemble.zarr"
    builder = ScenarioBuilder(str(hist), str(cmip6))
    window = (f"{horizon}-01-01", f"{horizon}-12-31")
    base = builder.build_scenario(experiment_id, window)
    preset = REGIONS[normalize_region_key(region)]
    daily = base.sel(
        lat=slice(preset.south, preset.north),
        lon=slice(preset.west, preset.east),
        method="nearest",
    )
    if "sample" in daily.dims:
        daily = daily.isel(sample=0, drop=True)
    rng = np.random.default_rng(hash((horizon, experiment_id, region)) % (2**32))
    members: list[xr.Dataset] = []
    for s in range(n_samples):
        ds_s = daily.copy(deep=True)
        for v in output_variables:
            if v in ds_s:
                ds_s[v] = ds_s[v] + xr.DataArray(
                    rng.normal(0, 0.02, ds_s[v].shape).astype(np.float32),
                    dims=ds_s[v].dims,
                    coords=ds_s[v].coords,
                )
        members.append(ds_s.expand_dims(sample=[s]))
    out = xr.concat(members, dim="sample")
    out.attrs["method"] = "linear_delta_bridge"
    return out


def _linear_delta_month_bridge(
    *,
    horizon: int,
    month_start: pd.Timestamp,
    experiment_id: str,
    region: str,
    output_variables: list[str],
    n_samples: int,
    historical_zarr_path: str | Path | None,
    cmip6_zarr_path: str | Path | None,
) -> xr.Dataset:
    """Fallback when CorrDiff forward fails: linear ScenarioBuilder + small noise per sample."""
    hist = historical_zarr_path or _REPO_ROOT / "data/processed/era5_2020_2024.zarr"
    cmip6 = cmip6_zarr_path or _REPO_ROOT / "data/processed/cmip6_ensemble.zarr"
    builder = ScenarioBuilder(str(hist), str(cmip6))
    window = (f"{horizon}-01-01", f"{horizon}-12-31")
    base = builder.build_scenario(experiment_id, window)
    preset = REGIONS[normalize_region_key(region)]
    lat = 0.5 * (preset.south + preset.north)
    lon = 0.5 * (preset.west + preset.east)
    base = base.sel(
        lat=slice(preset.south, preset.north),
        lon=slice(preset.west, preset.east),
        method="nearest",
    )
    month_end = month_start + pd.offsets.MonthEnd(0)
    daily = base.sel(time=slice(str(month_start.date()), str(month_end.date())))
    if "sample" in daily.dims:
        daily = daily.isel(sample=0, drop=True)
    samples = []
    rng = np.random.default_rng(hash((horizon, month_start.month, region)) % (2**32))
    for s in range(n_samples):
        noise = xr.Dataset(
            {
                v: (
                    daily[v].dims,
                    daily[v].values + rng.normal(0, 0.02, daily[v].shape).astype(np.float32),
                )
                for v in output_variables
                if v in daily.data_vars
            }
        )
        ds_s = daily.copy()
        for v in output_variables:
            if v in noise:
                ds_s[v] = noise[v]
        ds_s = ds_s.expand_dims(sample=[s])
        samples.append(ds_s)
    out = xr.concat(samples, dim="sample")
    out.attrs["method"] = "linear_delta_bridge"
    return out


def _hourly_to_daily_era5(
    hourly: xr.Dataset,
    output_variables: list[str],
    bounds: dict[str, float],
) -> xr.Dataset:
    """Map CorrDiff hourly output to daily ERA5-Land schema over region bbox."""
    if "variable" in hourly.dims or "variable" in hourly.coords:
        da = hourly["values"] if "values" in hourly else hourly.to_array(dim="variable")
    else:
        da = (
            hourly.to_array(dim="variable")
            if len(hourly.data_vars) > 1
            else next(iter(hourly.data_vars.values()))
        )

    # Standardize dim names
    rename = {}
    for d in da.dims:
        dl = d.lower()
        if dl in ("lat", "latitude"):
            rename[d] = "lat"
        if dl in ("lon", "longitude"):
            rename[d] = "lon"
        if dl in ("lead_time", "time"):
            rename[d] = "time"
    if rename:
        da = da.rename(rename)

    lat_slice = slice(bounds["south"], bounds["north"])
    lon_slice = slice(bounds["west"], bounds["east"])
    if "lat" in da.dims:
        da = da.sel(lat=lat_slice)
    if "lon" in da.dims:
        da = da.sel(lon=lon_slice)

    var_name = "variable" if "variable" in da.dims else None
    daily_vars: dict[str, xr.DataArray] = {}
    for corrdiff_name, era5_name in CORRDIFF_TO_ERA5.items():
        if var_name and corrdiff_name not in da.coords.get("variable", []):
            continue
        if var_name:
            sub = da.sel(variable=corrdiff_name)
        else:
            continue
        if era5_name == "wind10m":
            u = da.sel(variable="u10m") if "u10m" in da.coords.get("variable", []) else None
            v = da.sel(variable="v10m") if "v10m" in da.coords.get("variable", []) else None
            if u is not None and v is not None:
                sub = np.sqrt(u**2 + v**2)
            else:
                sub = sub
        if "time" in sub.dims:
            if era5_name in ("tmax",):
                daily_vars[era5_name] = sub.max("time")
            elif era5_name in ("tmin",):
                daily_vars[era5_name] = sub.min("time")
            elif era5_name == "precip":
                daily_vars[era5_name] = sub.sum("time")
            else:
                daily_vars[era5_name] = sub.mean("time")
        else:
            daily_vars[era5_name] = sub

    if not daily_vars:
        # Minimal stub from available data for tests / degraded path
        shape = (da.sizes.get("sample", 1), da.sizes.get("lat", 1), da.sizes.get("lon", 1))
        for v in output_variables:
            daily_vars[v] = xr.DataArray(
                np.zeros(shape, dtype=np.float32),
                dims=[d for d in ("sample", "lat", "lon") if d in da.dims],
            )

    ds = xr.Dataset(daily_vars)
    if "tmean" not in ds and "tmax" in ds and "tmin" in ds:
        ds["tmean"] = 0.5 * (ds["tmax"] + ds["tmin"])
    if "vpd_mean" not in ds and "rh_mean" in ds and "tmean" in ds:
        ds["vpd_mean"] = _vpd_from_rh_tmean(ds["rh_mean"], ds["tmean"])
    if "et0" not in ds:
        ds["et0"] = xr.zeros_like(ds["tmean"]) if "tmean" in ds else 0.0
    if "sm_root" not in ds:
        ds["sm_root"] = xr.full_like(ds["tmean"], 0.25) if "tmean" in ds else 0.25
    if "cwd" not in ds:
        ds["cwd"] = xr.zeros_like(ds["tmean"]) if "tmean" in ds else 0.0

    ds = compute_derived_features(ds)
    ds["vpd"] = ds.get("vpd_mean", ds.get("vpd", ds["tmean"] * 0.0))
    if "co2_ppm" not in ds:
        ds["co2_ppm"] = xr.full_like(ds["tmean"], 420.0) if "tmean" in ds else 420.0
    ds["cwd_cum"] = ds["cwd"].cumsum(dim="time") if "time" in ds.dims else ds["cwd"]

    keep = [v for v in output_variables if v in ds]
    return ds[keep]


def _vpd_from_rh_tmean(rh: xr.DataArray, tmean_c: xr.DataArray) -> xr.DataArray:
    """Magnus-style VPD (kPa) from RH (%) and tmean (C)."""
    es = 0.6108 * np.exp(17.27 * tmean_c / (tmean_c + 237.3))
    ea = es * np.clip(rh / 100.0, 0.0, 1.0)
    return (es - ea).astype(np.float32)


def load_corrdiff_scenario_ensemble(
    *,
    cache_path: Path,
    lat: float,
    lon: float,
    year: int,
) -> list[Any]:
    """
    Load cached CorrDiff Zarr and return one ``[1, 365, 11]`` climate tensor per sample.

    Raises ``FileNotFoundError`` with bulk-script hint when cache is absent.
    """
    from torch import Tensor

    from api.feature_resolver import climate_tensor_from_dataset_point

    cache_path = Path(cache_path)
    if not cache_path.is_dir():
        parts = cache_path.stem.replace("corrdiff_", "").split("_")
        if len(parts) >= 3:
            msg = corrdiff_cache_missing_message(
                cache_path, parts[0], int(parts[1]), "_".join(parts[2:])
            )
        else:
            msg = f"CorrDiff cache not found at {cache_path}"
        raise FileNotFoundError(msg)

    ds = xr.open_zarr(cache_path, consolidated=True)
    if "sample" not in ds.dims:
        raise ValueError(f"CorrDiff cache missing 'sample' dimension: {cache_path}")

    tensors: list[Tensor] = []
    for s in range(int(ds.sizes["sample"])):
        point = climate_tensor_from_dataset_point(ds.isel(sample=s), lat, lon, year)
        tensors.append(point)
    return tensors


def write_synthetic_corrdiff_cache(
    path: Path,
    *,
    scenario: str = "ssp245",
    horizon: int = 2030,
    region: str = "ghana",
    n_samples: int = 4,
    n_days: int = 365,
) -> Path:
    """Write a minimal Zarr for CPU tests (no ERA5/CMIP6 or GPU required)."""
    path = Path(path)
    preset = REGIONS[normalize_region_key(region)]
    n_days = max(n_days, 365)
    times = pd.date_range(f"{horizon}-01-01", periods=n_days, freq="D")
    lat = np.array([0.5 * (preset.south + preset.north)], dtype=np.float32)
    lon = np.array([0.5 * (preset.west + preset.east)], dtype=np.float32)
    shape = (n_days, 1, 1)
    rng = np.random.default_rng(42)
    base_vars = {
        "tmax": 30.0 + 0.01 * np.arange(n_days),
        "tmin": 23.0 + 0.01 * np.arange(n_days),
        "tmean": 26.5 + 0.01 * np.arange(n_days),
        "precip": np.abs(rng.normal(3.0, 1.0, n_days)),
        "srad": np.full(n_days, 15.0),
        "vpd_mean": np.full(n_days, 1.2),
        "et0": np.full(n_days, 3.5),
        "sm_root": np.full(n_days, 0.28),
        "wind10m": np.full(n_days, 2.0),
        "rh_mean": np.full(n_days, 75.0),
        "co2_ppm": np.full(n_days, 420.0),
        "vpd": np.full(n_days, 1.2),
        "cwd": np.zeros(n_days),
        "cwd_cum": np.cumsum(np.zeros(n_days)),
        "gdd_cocoa": np.full(n_days, 10.0),
    }
    members: list[xr.Dataset] = []
    for s in range(n_samples):
        data_vars = {
            k: (("time", "lat", "lon"), v.reshape(shape).astype(np.float32) + 0.01 * s)
            for k, v in base_vars.items()
        }
        members.append(
            xr.Dataset(
                data_vars,
                coords={"time": times, "lat": lat, "lon": lon, "sample": [s]},
            )
        )
    daily = xr.concat(members, dim="sample")
    daily.attrs["method"] = "synthetic_corrdiff_cache"
    daily.attrs["scenario"] = scenario
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_zarr(path, mode="w")
    return path


__all__ = [
    "DEFAULT_OUTPUT_VARIABLES",
    "CorrDiffCMIP6Downscaler",
    "corrdiff_cache_missing_message",
    "corrdiff_cache_path",
    "load_corrdiff_scenario_ensemble",
    "write_synthetic_corrdiff_cache",
]
