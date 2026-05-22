"""
Delta-method downscaling of 0.5° ATTRICI counterfactual climate to the ERA5-Land grid.

Rationale: rerunning ATTRICI at ~9 km would be ~500× more expensive; ISIMIP3a
counterfactual methodology was validated at 0.5°. The delta method is the standard
ISIMIP3b approach for higher-resolution bias-adjusted products (cf. ISIMIP3BASD;
Lange 2019).
"""

from __future__ import annotations

import structlog

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from data.era5_ingest import (
    FAO_ALBEDO,
    FAO_GAMMA,
    KELVIN_OFFSET,
    MAGNUS_A,
    MAGNUS_B,
    MAGNUS_C,
    WIND10_TO_WIND2_FACTOR,
    compute_derived_features,
)

log = structlog.get_logger(__name__)

_KELVIN_THRESHOLD = 150.0

# ISIMIP short name → ERA5-Land feature name (``era5_ingest.OUTPUT_VARS``)
ISIMIP_TO_ERA5: dict[str, str] = {
    "tas": "tmean",
    "tasmin": "tmin",
    "tasmax": "tmax",
    "pr": "precip",
    "hurs": "rh_mean",
    "rsds": "srad",
    "sfcwind": "wind10m",
}

_TEMPERATURE_ISIMIP = frozenset({"tas", "tasmin", "tasmax"})

# Variables adjusted on the 9 km grid (``hurs`` optional if present at 0.5°)
DEFAULT_ADDITIVE_ISIMIP: tuple[str, ...] = ("tas", "tasmin", "tasmax", "hurs")
DEFAULT_MULTIPLICATIVE_ISIMIP: tuple[str, ...] = ("pr", "rsds", "sfcwind")

_DERIVED_FROM_COMPUTE_FEATURES = frozenset(
    {
        "gdd_cocoa",
        "heat_days_above_32c",
        "dry_spell_max",
        "vpd_mean_30d",
        "vpd_mean_90d",
        "cwd_30d",
        "cwd_90d",
        "sm_root_30d",
        "sm_root_90d",
    }
)


@dataclass
class DeltaDownscaler:
    """
    Bridge 0.5° factual / counterfactual ISIMIP fields to ERA5-Land daily features.

    Workflow
    --------
    1. Monthly climatology of factual and counterfactual 0.5° fields.
    2. Additive delta (temperature, humidity) or multiplicative ratio (pr, rsds, wind).
    3. Bilinear interpolation to the 9 km target grid.
    4. Apply to factual ERA5-Land; recompute VPD, ET0, CWD; then cocoa derived features.
    """

    factual_05deg: xr.Dataset
    counterfactual_05deg: xr.Dataset
    additive_isimip: tuple[str, ...] = DEFAULT_ADDITIVE_ISIMIP
    multiplicative_isimip: tuple[str, ...] = DEFAULT_MULTIPLICATIVE_ISIMIP
    min_ratio: float = 1e-6
    _delta_additive: xr.Dataset | None = field(default=None, init=False, repr=False)
    _ratio_multiplicative: xr.Dataset | None = field(default=None, init=False, repr=False)

    def build_delta(self) -> xr.Dataset:
        """
        Compute monthly climatology deltas / ratios on the 0.5° grid.

        Returns
        -------
        xarray.Dataset
            Contains additive deltas and multiplicative ratios (``month``, ``lat``,
            ``lon``), with ISIMIP variable names.
        """
        factual = _harmonize_05deg(self.factual_05deg)
        counter = _harmonize_05deg(self.counterfactual_05deg)

        f_clim = _monthly_climatology(factual)
        c_clim = _monthly_climatology(counter)

        additive_vars: dict[str, xr.DataArray] = {}
        for var in self.additive_isimip:
            if var in f_clim and var in c_clim:
                additive_vars[var] = (f_clim[var] - c_clim[var]).astype(np.float32)

        ratio_vars: dict[str, xr.DataArray] = {}
        for var in self.multiplicative_isimip:
            if var in f_clim and var in c_clim:
                denom = c_clim[var].where(c_clim[var] > 0)
                ratio = (f_clim[var] / denom).where(denom > 0)
                ratio_vars[var] = ratio.clip(min=self.min_ratio).astype(np.float32)

        self._delta_additive = xr.Dataset(additive_vars) if additive_vars else xr.Dataset()
        self._ratio_multiplicative = xr.Dataset(ratio_vars) if ratio_vars else xr.Dataset()

        combined = xr.merge([self._delta_additive, self._ratio_multiplicative], compat="override")
        combined.attrs["method_additive"] = list(additive_vars.keys())
        combined.attrs["method_multiplicative"] = list(ratio_vars.keys())
        log.info(
            "Built delta: %d additive, %d multiplicative variables",
            len(additive_vars),
            len(ratio_vars),
        )
        return combined

    def apply_to_factual(self, ds_9km: xr.Dataset) -> xr.Dataset:
        """
        Apply interpolated deltas to factual ERA5-Land and refresh derived features.

        Parameters
        ----------
        ds_9km:
            Factual daily ERA5-Land feature cube (e.g. from :class:`data.era5_ingest.ERA5Ingest`).

        Returns
        -------
        xarray.Dataset
            Counterfactual 9 km dataset with recomputed VPD, ET0, CWD, and cocoa features.
        """
        if self._delta_additive is None and self._ratio_multiplicative is None:
            self.build_delta()

        out = _harmonize_9km(ds_9km).copy(deep=True)
        lat_name, lon_name = _lat_lon_names(out)

        for isimip_var, era5_var in ISIMIP_TO_ERA5.items():
            if era5_var not in out:
                continue
            if (
                self._delta_additive is not None
                and isimip_var in self._delta_additive
            ):
                delta_9 = _interp_monthly_to_grid(
                    self._delta_additive[isimip_var], out, lat_name, lon_name
                )
                out[era5_var] = _apply_monthly_additive(out[era5_var], delta_9)
            elif (
                self._ratio_multiplicative is not None
                and isimip_var in self._ratio_multiplicative
            ):
                ratio_9 = _interp_monthly_to_grid(
                    self._ratio_multiplicative[isimip_var], out, lat_name, lon_name
                )
                out[era5_var] = _apply_monthly_multiplicative(out[era5_var], ratio_9)

        out = _recompute_vpd_et0_cwd(out)
        if "cwd_cum" in ds_9km:
            out["cwd_cum"] = out["cwd"].cumsum(dim="time")

        drop = [v for v in out.data_vars if v in _DERIVED_FROM_COMPUTE_FEATURES]
        if drop:
            out = out.drop_vars(drop, errors="ignore")

        out = compute_derived_features(out)
        out.attrs.update(
            {
                "downscaling_method": "delta",
                "delta_source_resolution_deg": 0.5,
                "reference": "ISIMIP3b delta method; Lange 2019 ISIMIP3BASD",
            }
        )
        return out


def _lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.dims or "latitude" in ds.coords:
        return "latitude", "longitude"
    return "lat", "lon"


def _rename_coords_05deg(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if "lon" in ds.coords and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    return ds


def _harmonize_05deg(ds: xr.Dataset) -> xr.Dataset:
    ds = _rename_coords_05deg(ds)
    out_vars: dict[str, xr.DataArray] = {}
    for var in ds.data_vars:
        da = ds[var]
        if var in _TEMPERATURE_ISIMIP and float(da.mean(skipna=True)) > _KELVIN_THRESHOLD:
            da = da - KELVIN_OFFSET
        out_vars[var] = da
    return xr.Dataset(out_vars, coords=ds.coords, attrs=ds.attrs)


def _harmonize_9km(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def _monthly_climatology(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.dims:
        raise ValueError("Dataset must have a time dimension for monthly climatology")
    return ds.groupby("time.month").mean(dim="time", skipna=True)


def _interp_monthly_to_grid(
    monthly: xr.DataArray,
    target: xr.Dataset,
    lat_name: str,
    lon_name: str,
) -> xr.DataArray:
    """Bilinear interpolation of monthly fields onto the ERA5-Land grid."""
    da = _rename_coords_05deg(monthly.to_dataset(name=monthly.name))[monthly.name]
    interp = da.interp(
        lat=target[lat_name],
        lon=target[lon_name],
        method="linear",
    )
    rename_map: dict[str, str] = {}
    if "lat" in interp.dims and lat_name != "lat":
        rename_map["lat"] = lat_name
    if "lon" in interp.dims and lon_name != "lon":
        rename_map["lon"] = lon_name
    if rename_map:
        interp = interp.rename(rename_map)
    return interp


def _apply_monthly_additive(da: xr.DataArray, delta: xr.DataArray) -> xr.DataArray:
    return da.groupby("time.month") - delta


def _apply_monthly_multiplicative(da: xr.DataArray, ratio: xr.DataArray) -> xr.DataArray:
    adjusted = da.groupby("time.month") / ratio
    return adjusted.where(ratio > 0)


def _saturation_vapor_pressure_kpa_array(tmean_c: xr.DataArray) -> xr.DataArray:
    return MAGNUS_A * np.exp(MAGNUS_B * tmean_c / (MAGNUS_C + tmean_c))


def _recompute_vpd_et0_cwd(ds: xr.Dataset) -> xr.Dataset:
    """Recompute VPD, reference ET0, and CWD from adjusted daily drivers."""
    out = ds.copy()
    tmean = out["tmean"]
    rh = out["rh_mean"].clip(0, 100)
    es = _saturation_vapor_pressure_kpa_array(tmean)
    out["vpd_mean"] = (es * (1.0 - rh / 100.0)).clip(min=0)

    wind10m = out["wind10m"]
    srad = out["srad"]
    u2 = wind10m * WIND10_TO_WIND2_FACTOR
    rn = srad * (1.0 - FAO_ALBEDO)
    t_k = tmean + KELVIN_OFFSET
    delta_slope = es * MAGNUS_B * MAGNUS_C / (tmean + MAGNUS_C) ** 2
    vpd = out["vpd_mean"]
    num_rad = delta_slope * rn * 0.408
    num_aero = FAO_GAMMA * (900.0 / t_k) * u2 * vpd
    den = delta_slope + FAO_GAMMA * (1.0 + 0.34 * u2)
    out["et0"] = (num_rad + num_aero) / den
    out["et0"] = out["et0"].clip(min=0)
    out["cwd"] = out["et0"] - out["precip"]
    return out
