"""
Build future daily climate scenarios by applying CMIP6 delta-change factors to ERA5-Land.

This is intentionally compatible with :mod:`counterfactual.delta_downscaler` utilities:
- additive deltas for temperature / humidity
- multiplicative ratios for precipitation / radiation / wind
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from counterfactual.delta_downscaler import (
    _apply_monthly_additive,
    _apply_monthly_multiplicative,
    _recompute_vpd_et0_cwd,
)
from api.telemetry import trace_span
from data.era5_ingest import compute_derived_features


@dataclass
class ScenarioBuilder:
    historical_zarr_path: str
    cmip6_zarr_path: str

    def _open_hist(self) -> xr.Dataset:
        return xr.open_zarr(self.historical_zarr_path)

    def _open_cmip6(self) -> xr.Dataset:
        return xr.open_zarr(self.cmip6_zarr_path)

    def build_scenario(self, scenario: str, window: tuple[str, str]) -> xr.Dataset:
        with trace_span("scenario_builder.build", scenario=scenario):
            return self._build_scenario_impl(scenario, window)

    def _build_scenario_impl(self, scenario: str, window: tuple[str, str]) -> xr.Dataset:
        """
        Apply delta-change from CMIP6 (raw vars) to historical ERA5 schema vars.

        Parameters
        ----------
        scenario:
            SSP scenario string (e.g. ``ssp245``).
        window:
            (start, end) date strings used to compute future climatology.
        """
        hist = self._open_hist()
        fut = self._open_cmip6().sel(scenario=scenario, time=slice(window[0], window[1]))

        # Build monthly climatologies for raw CMIP6 vars, aggregated over model.
        fut_m = fut.mean(dim="model").groupby("time.month").mean("time")

        # Convert historical ERA5 schema to raw CMIP6-like units for baseline climatology.
        # - temperatures: C → K
        # - precip: mm/day → kg m-2 s-1
        # - srad: MJ/m2/day → W/m2
        hist_raw = xr.Dataset(
            {
                "tas": hist["tmean"] + 273.15,
                "tasmax": hist["tmax"] + 273.15,
                "tasmin": hist["tmin"] + 273.15,
                "hurs": hist["rh_mean"],
                "pr": hist["precip"] / 86400.0,
                "rsds": hist["srad"] * 1e6 / 86400.0,
                "sfcWind": hist["wind10m"],
            }
        )
        hist_m = hist_raw.groupby("time.month").mean("time")

        # Compute deltas/ratios (monthly, lat/lon)
        # delta_downscaler helpers are defined as: adjusted = factual - delta.
        # For forward-looking scenarios we want: scenario = historical + (future - historical).
        # So we pass the negated delta.
        delta_tas = (hist_m["tas"] - fut_m["tas"]).astype(np.float32)
        delta_tasmax = (hist_m["tasmax"] - fut_m["tasmax"]).astype(np.float32)
        delta_tasmin = (hist_m["tasmin"] - fut_m["tasmin"]).astype(np.float32)
        delta_hurs = (hist_m["hurs"] - fut_m["hurs"]).astype(np.float32)

        # multiplicative helper is adjusted = factual / ratio; to apply forward ratio we invert.
        ratio_pr = (hist_m["pr"] / fut_m["pr"].where(fut_m["pr"] != 0)).clip(min=1e-6).astype(
            np.float32
        )
        ratio_rsds = (
            hist_m["rsds"] / fut_m["rsds"].where(fut_m["rsds"] != 0)
        ).clip(min=1e-6).astype(np.float32)
        ratio_wind = (
            hist_m["sfcWind"] / fut_m["sfcWind"].where(fut_m["sfcWind"] != 0)
        ).clip(min=1e-6).astype(np.float32)

        out = hist.copy()
        out["tmean"] = _apply_monthly_additive(out["tmean"], delta_tas)
        out["tmax"] = _apply_monthly_additive(out["tmax"], delta_tasmax)
        out["tmin"] = _apply_monthly_additive(out["tmin"], delta_tasmin)
        out["rh_mean"] = _apply_monthly_additive(out["rh_mean"], delta_hurs)
        out["precip"] = _apply_monthly_multiplicative(out["precip"], ratio_pr)
        out["srad"] = _apply_monthly_multiplicative(out["srad"], ratio_rsds)
        out["wind10m"] = _apply_monthly_multiplicative(out["wind10m"], ratio_wind)

        out = _recompute_vpd_et0_cwd(out)
        out["cwd_cum"] = out["cwd"].cumsum(dim="time")

        # Ensure yield-surrogate compatibility: add vpd alias and co2 channel.
        out["vpd"] = out["vpd_mean"]
        if "co2_ppm" not in out:
            out["co2_ppm"] = xr.full_like(out["tmean"], 420.0)

        out = compute_derived_features(out)
        out.attrs["scenario"] = scenario
        out.attrs["window_start"] = window[0]
        out.attrs["window_end"] = window[1]
        out.attrs["method"] = "cmip6_delta_change"
        return out

