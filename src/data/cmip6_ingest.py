"""
CMIP6 future-climate ingest via Google Earth Engine (NASA GDDP-CMIP6).

Collection:
    ``NASA/GDDP-CMIP6`` — daily, ~0.25°, bias-corrected and statistically downscaled.

This module parallels :mod:`data.era5_ingest` but produces an ensemble cube across
GCM models and SSP scenarios, with derived variables aligned to the ERA5 ingest schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import ee
import numpy as np
import xarray as xr

# Registers the ``ee`` Xarray backend (Xee).
import xee  # noqa: F401

from data.agromet import KELVIN_OFFSET, fao_et0_daily, magnus_es_kpa
from data.gee_auth import initialize_earth_engine

CMIP6_GDDP_DAILY = "NASA/GDDP-CMIP6"

DEFAULT_MODELS: list[str] = [
    "ACCESS-CM2",
    "CESM2",
    "CNRM-ESM2-1",
    "EC-Earth3",
    "GFDL-ESM4",
    "INM-CM5-0",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "NorESM2-MM",
]

DEFAULT_SCENARIOS: list[str] = ["ssp126", "ssp245", "ssp370", "ssp585"]

DEFAULT_VARIABLES: list[str] = ["tas", "tasmax", "tasmin", "pr", "hurs", "rsds", "sfcWind"]

OUTPUT_VARS_ERA5_SCHEMA: tuple[str, ...] = (
    "tmax",
    "tmin",
    "tmean",
    "rh_mean",
    "vpd_mean",
    "precip",
    "et0",
    "cwd",
    "cwd_cum",
    "sm_root",
    "wind10m",
    "srad",
)

# Raw CMIP6 variable names (as in GEE collection)
RAW_VARS: tuple[str, ...] = ("tas", "tasmax", "tasmin", "pr", "hurs", "rsds", "sfcWind")


def _to_celsius(img: ee.Image, band: str, name: str) -> ee.Image:
    return img.select(band).subtract(KELVIN_OFFSET).rename(name)


def _pr_to_mm_day(pr: ee.Image) -> ee.Image:
    # GDDP-CMIP6 ``pr`` is typically kg m-2 s-1; 1 kg m-2 = 1 mm water.
    return pr.multiply(86400.0).rename("precip")


def _rsds_to_mj_day(rsds: ee.Image) -> ee.Image:
    # rsds in W m-2 → MJ m-2 day-1
    return rsds.multiply(86400.0).divide(1e6).rename("srad")


def _build_daily_ic(
    aoi: ee.Geometry,
    start: str,
    end: str,
    *,
    model: str,
    scenario: str,
    variables: Iterable[str],
) -> ee.ImageCollection:
    ic = (
        ee.ImageCollection(CMIP6_GDDP_DAILY)
        .filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("model", model))
        .filter(ee.Filter.eq("scenario", scenario))
        .select(list(variables))
    )

    def _enrich(img: ee.Image) -> ee.Image:
        # Temperatures
        tmean = _to_celsius(img, "tas", "tmean")
        tmax = _to_celsius(img, "tasmax", "tmax")
        tmin = _to_celsius(img, "tasmin", "tmin")

        # RH (%)
        rh = img.select("hurs").rename("rh_mean").clamp(0, 100)

        # VPD (kPa) from Magnus
        es = magnus_es_kpa(tmean)
        vpd = es.multiply(ee.Image(1).subtract(rh.divide(100.0))).rename("vpd_mean")

        # Forcings
        wind10m = img.select("sfcWind").rename("wind10m")
        srad = _rsds_to_mj_day(img.select("rsds"))
        precip = _pr_to_mm_day(img.select("pr"))

        # ET0 + CWD
        et0 = fao_et0_daily(tmean, rh, wind10m, srad)
        cwd = et0.subtract(precip).rename("cwd")

        # Soil moisture is not available in this collection; keep placeholder constant
        # to match the ERA5 schema (downstream PINN expects a channel).
        sm_root = ee.Image.constant(0.28).rename("sm_root")

        raw = img.select(list(variables))
        return (
            ee.Image.cat(
                [
                    raw,
                    tmax,
                    tmin,
                    tmean,
                    rh,
                    vpd,
                    precip,
                    et0,
                    cwd,
                    sm_root,
                    wind10m,
                    srad,
                ]
            )
            .copyProperties(img, ["system:time_start"])
            .clip(aoi)
        )

    return ic.map(_enrich)


@dataclass
class CMIP6Ingest:
    """
    Ingest daily CMIP6 downscaled fields for an AOI and build ensemble datasets.
    """

    aoi: ee.Geometry
    start: str
    end: str
    models: list[str] | None = None
    scenarios: list[str] | None = None
    variables: list[str] | None = None
    chunks: dict[str, int] | None = None
    project: str | None = None
    scale: int = 25_000  # ~0.25°

    def __post_init__(self) -> None:
        self.models = list(self.models) if self.models is not None else list(DEFAULT_MODELS)
        self.scenarios = (
            list(self.scenarios) if self.scenarios is not None else list(DEFAULT_SCENARIOS)
        )
        self.variables = (
            list(self.variables) if self.variables is not None else list(DEFAULT_VARIABLES)
        )
        self.chunks = self.chunks or {"time": 30, "latitude": 128, "longitude": 128}

    def _open_one(self, *, model: str, scenario: str) -> xr.Dataset:
        ic = _build_daily_ic(
            self.aoi,
            self.start,
            self.end,
            model=model,
            scenario=scenario,
            variables=self.variables or DEFAULT_VARIABLES,
        )
        ds = xr.open_dataset(
            ic,
            engine="ee",
            geometry=self.aoi,
            scale=self.scale,
            chunks=self.chunks,
        )
        rename: dict[str, str] = {}
        if "lat" in ds.dims:
            rename["lat"] = "latitude"
        if "lon" in ds.dims:
            rename["lon"] = "longitude"
        if rename:
            ds = ds.rename(rename)
        keep = [
            v
            for v in (list(RAW_VARS) + [x for x in OUTPUT_VARS_ERA5_SCHEMA if x != "cwd_cum"])
            if v in ds.data_vars
        ]
        ds = ds[keep]
        ds["cwd_cum"] = ds["cwd"].cumsum(dim="time")
        ds.attrs.update(
            {
                "source": "Google Earth Engine",
                "collection": CMIP6_GDDP_DAILY,
                "model": model,
                "scenario": scenario,
                "start_date": self.start,
                "end_date": self.end,
                "schema": "era5_compatible_daily",
            }
        )
        return ds

    def build_ensemble(self) -> xr.Dataset:
        """
        Returns an ensemble dataset with dims (time, latitude, longitude, model, scenario).
        """
        initialize_earth_engine(project=self.project)

        models = self.models or []
        scenarios = self.scenarios or []
        if not models or not scenarios:
            raise ValueError("No models/scenarios requested")

        by_model: list[xr.Dataset] = []
        for m in models:
            by_scenario: list[xr.Dataset] = []
            for s in scenarios:
                ds = self._open_one(model=m, scenario=s).expand_dims(scenario=[s])
                by_scenario.append(ds)
            by_model.append(xr.concat(by_scenario, dim="scenario").expand_dims(model=[m]))

        out = xr.concat(by_model, dim="model")
        return out.transpose("time", "latitude", "longitude", "model", "scenario")

    def ensemble_statistics(self, ds: xr.Dataset | None = None) -> xr.Dataset:
        """
        Compute model-mean and p10/p50/p90 across model dimension, per scenario.
        """
        work = ds if ds is not None else self.build_ensemble()
        mean = work.mean(dim="model")
        q = work.quantile([0.1, 0.5, 0.9], dim="model")
        q = q.rename({"quantile": "stat"})
        q = q.assign_coords(stat=["p10", "p50", "p90"])
        mean = mean.expand_dims(stat=["mean"])
        return xr.concat([mean, q], dim="stat")

    def delta_change_factors(
        self,
        historical_ds: xr.Dataset,
        future_window: tuple[str, str],
        *,
        additive: tuple[str, ...] = ("tas", "tasmin", "tasmax", "hurs"),
        multiplicative: tuple[str, ...] = ("pr", "rsds", "sfcWind"),
    ) -> xr.Dataset:
        """
        Monthly delta factors versus historical baseline climatology.

        Returns a dataset with ``month`` coordinate and variable names matching ISIMIP/CMIP6.
        Additive vars are returned as deltas; multiplicative vars as ratios.
        """
        future = self.build_ensemble().sel(time=slice(future_window[0], future_window[1]))
        # Historical baseline climatology from provided dataset
        hist = historical_ds

        def _monthly_clim(ds0: xr.Dataset) -> xr.Dataset:
            return ds0.groupby("time.month").mean("time")

        f_clim = _monthly_clim(future)
        h_clim = _monthly_clim(hist)

        out_vars: dict[str, xr.DataArray] = {}
        for v in additive:
            if v in f_clim and v in h_clim:
                out_vars[v] = (f_clim[v] - h_clim[v]).astype(np.float32)
        for v in multiplicative:
            if v in f_clim and v in h_clim:
                denom = h_clim[v].where(h_clim[v] != 0)
                out_vars[v] = (f_clim[v] / denom).clip(min=1e-6).astype(np.float32)

        out = xr.Dataset(out_vars)
        out.attrs["additive_vars"] = list(additive)
        out.attrs["multiplicative_vars"] = list(multiplicative)
        return out

