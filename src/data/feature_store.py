"""
Unified feature store for PINN-ready tensors (climate + static).

This module joins:
- ERA5 daily climate (Zarr from :mod:`data.era5_ingest`)
- SoilGrids/iSDA static cube (Zarr from :mod:`data.soil_ingest`)
- GEDI canopy summaries (Zarr from :mod:`data.gedi_ingest`)
- SMAP soil moisture (Zarr from :mod:`data.smap_ingest`)

The output matches :class:`~models.yield_surrogate.YieldSurrogateModel` expectations:
- climate: [B, T, 11] in CLIMATE_CHANNEL_NAMES order
- static: [B, F] where index 0 is AWC (mm); additional features are optional and
  controlled by ``static_feature_names``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import Tensor

from models.yield_surrogate import CLIMATE_CHANNEL_NAMES, STATIC_FEATURE_NAMES


_ZARR_CLIMATE_ALIASES: dict[str, tuple[str, ...]] = {
    "tmax": ("tmax",),
    "tmin": ("tmin",),
    "tmean": ("tmean",),
    "precip": ("precip", "pr"),
    "srad": ("srad", "ssrd"),
    "vpd": ("vpd", "vpd_mean"),
    "et0": ("et0",),
    "sm_root": ("sm_root", "swvl1"),
    "wind10m": ("wind10m", "u10"),
    "rh_mean": ("rh_mean", "rh"),
    "co2_ppm": ("co2_ppm", "co2"),
}


def _lat_lon_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.coords:
        return "latitude", "longitude"
    if "lat" in ds.coords:
        return "lat", "lon"
    raise ValueError(f"No lat/lon coordinates in dataset: {list(ds.coords)}")


def _pick_var(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.data_vars:
            return name
    raise KeyError(f"None of {names} in dataset vars {list(ds.data_vars)}")


def _climate_tensor_from_dataset(ds: xr.Dataset, year: int, *, sequence_length: int = 365) -> np.ndarray:
    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError("Climate dataset missing time dimension")

    annual = ds.sel(time=ds["time"].dt.year == year)
    if int(annual.sizes.get("time", 0)) < sequence_length:
        annual = ds.sortby("time").isel(time=slice(-sequence_length, None))
    else:
        annual = annual.isel(time=slice(0, sequence_length))

    channels: list[np.ndarray] = []
    for name in CLIMATE_CHANNEL_NAMES:
        var = _pick_var(annual, _ZARR_CLIMATE_ALIASES[name])
        values = np.asarray(annual[var].values, dtype=np.float32).reshape(-1)
        if values.size < sequence_length:
            pad = np.full(sequence_length - values.size, values[-1] if values.size else 0.0)
            values = np.concatenate([pad, values])
        elif values.size > sequence_length:
            values = values[-sequence_length:]
        channels.append(values)

    return np.stack(channels, axis=-1).astype(np.float32)


def _nearest_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    lat_name, lon_name = _lat_lon_coord_names(ds)
    return ds.sel({lat_name: lat, lon_name: lon}, method="nearest")


@dataclass(frozen=True)
class FeatureStorePaths:
    era5_zarr: Path
    soil_zarr: Path | None = None
    gedi_zarr: Path | None = None
    smap_zarr: Path | None = None


class FeatureStore:
    """
    Load and join climate/static modalities from Zarr stores.

    By default, looks for conventional file names under ``zarr_root``; you can also
    pass explicit paths via :class:`FeatureStorePaths`.
    """

    def __init__(
        self,
        zarr_root: str | Path,
        *,
        paths: FeatureStorePaths | None = None,
        consolidated: bool = True,
    ) -> None:
        root = Path(zarr_root)
        self.root = root
        if paths is None:
            paths = FeatureStorePaths(
                era5_zarr=root / "era5_2020_2024.zarr",
                soil_zarr=(root / "soilgrids_static.zarr") if (root / "soilgrids_static.zarr").is_dir() else None,
                gedi_zarr=(root / "gedi_monthly.zarr") if (root / "gedi_monthly.zarr").is_dir() else None,
                smap_zarr=(root / "smap_l4.zarr") if (root / "smap_l4.zarr").is_dir() else None,
            )
        self.paths = paths
        self.consolidated = consolidated

    def _open(self, path: Path) -> xr.Dataset:
        return xr.open_zarr(path, consolidated=self.consolidated)

    def climate_tensor(
        self,
        *,
        lat: float,
        lon: float,
        year: int,
        climate_window_days: int = 365,
    ) -> Tensor:
        ds = self._open(self.paths.era5_zarr)
        point = _nearest_point(ds, lat, lon)
        arr = _climate_tensor_from_dataset(point, year, sequence_length=climate_window_days)
        if arr.shape[0] != climate_window_days:
            arr = arr[-climate_window_days:]
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)

    def static_vector(
        self,
        *,
        lat: float,
        lon: float,
        static_feature_names: tuple[str, ...] | None = None,
    ) -> Tensor:
        """
        Build a static vector matching a model's `static_feature_names` (or default 10).

        Supported feature keys (subset):
        - ``awc_mm`` (required; index 0 in the legacy 10-feature layout)
        - ``canopy_height_p95`` (GEDI)
        - ``soc_0_5`` (SoilGrids)
        - ``clay_0_30`` (SoilGrids; average of 0-5,5-15,15-30)
        """
        names = static_feature_names or STATIC_FEATURE_NAMES
        vec = np.zeros(len(names), dtype=np.float32)

        # Soil features
        soil = None
        if self.paths.soil_zarr is not None and self.paths.soil_zarr.is_dir():
            soil = self._open(self.paths.soil_zarr)
            soil_pt = _nearest_point(soil, lat, lon)
            if "awc_mm" in soil_pt:
                awc_val = float(soil_pt["awc_mm"].values)
            elif "soil" in soil_pt and {"sand", "clay", "soc"}.issubset(set(soil_pt["soil"].coords.get("property", []))):
                awc_val = float(np.nan)
            else:
                awc_val = float(np.nan)
        else:
            awc_val = float(np.nan)

        # GEDI features
        canopy_p95 = float(np.nan)
        if self.paths.gedi_zarr is not None and self.paths.gedi_zarr.is_dir():
            gedi = self._open(self.paths.gedi_zarr)
            gedi_pt = _nearest_point(gedi, lat, lon)
            if "canopy_height_p95" in gedi_pt:
                canopy_p95 = float(gedi_pt["canopy_height_p95"].values)

        # Fill requested features
        for i, name in enumerate(names):
            if name == "awc_mm":
                vec[i] = np.nan_to_num(awc_val, nan=0.0)
            elif name == "canopy_height_p95":
                vec[i] = np.nan_to_num(canopy_p95, nan=0.0)
            else:
                vec[i] = 0.0

        return torch.from_numpy(vec).unsqueeze(0)

    def to_pinn_tensors(
        self,
        *,
        locations: list[tuple[float, float]],
        year: int,
        climate_window_days: int = 365,
        static_feature_names: tuple[str, ...] | None = None,
    ) -> tuple[Tensor, Tensor]:
        climates: list[Tensor] = []
        statics: list[Tensor] = []
        for lat, lon in locations:
            climates.append(
                self.climate_tensor(lat=lat, lon=lon, year=year, climate_window_days=climate_window_days)
            )
            statics.append(
                self.static_vector(lat=lat, lon=lon, static_feature_names=static_feature_names)
            )
        climate = torch.cat(climates, dim=0)
        static = torch.cat(statics, dim=0)
        if climate.shape[-1] != len(CLIMATE_CHANNEL_NAMES):
            raise ValueError("Climate tensor does not match CLIMATE_CHANNEL_NAMES width")
        return climate, static


__all__ = ["FeatureStore", "FeatureStorePaths"]

