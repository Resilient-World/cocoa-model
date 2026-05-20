"""
SoilGrids 2.0 + iSDA Africa soil covariate ingest (static).

This module provides a thin Earth Engine → lazy Xarray wrapper similar to other
ingest modules, but for soil properties by depth.

Primary use:
- Build a static soil cube with dims (latitude, longitude, property, depth)
- Derive available water capacity (AWC, mm) using Saxton–Rawls (2006) PTFs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ee
import numpy as np
import xarray as xr

# Registers the ``ee`` Xarray backend (Xee).
import xee  # noqa: F401

from data.gee_auth import initialize_earth_engine

# ---------------------------------------------------------------------------
# Assets + defaults
# ---------------------------------------------------------------------------

# SoilGrids 2.0 naming as requested (property-specific images)
SOILGRIDS_20_ASSET_TEMPLATE = "projects/soilgrids-isric/{property}_mean"

# iSDA Africa (gap fill), optional and only meaningful over Africa
ISDA_AFRICA_TEMPLATE = "ISDASOIL/Africa/v1/{property}_{depth}"

DEFAULT_PROPERTIES: tuple[str, ...] = (
    "clay",
    "sand",
    "silt",
    "soc",
    "phh2o",
    "cec",
    "bdod",
    "nitrogen",
)

DEFAULT_DEPTHS_CM: tuple[tuple[int, int], ...] = (
    (0, 5),
    (5, 15),
    (15, 30),
    (30, 60),
    (60, 100),
)

DEFAULT_SCALE_M = 250  # SoilGrids native resolution ~250 m


def _depth_label(depth_cm: tuple[int, int]) -> str:
    lo, hi = depth_cm
    return f"{lo}-{hi}"


def _soilgrids_band_for_depth(depth_cm: tuple[int, int]) -> str:
    # Most SoilGrids depth-coded images use e.g. "0-5cm_mean" band names.
    return f"{_depth_label(depth_cm)}cm_mean"


def _isda_asset(property_name: str, depth_cm: tuple[int, int]) -> str:
    return ISDA_AFRICA_TEMPLATE.format(property=property_name, depth=_depth_label(depth_cm))


def _saxton_rawls_theta1500(sand_pct: np.ndarray, clay_pct: np.ndarray, om_pct: np.ndarray) -> np.ndarray:
    """
    Volumetric water content at 1500 kPa (wilting point), Saxton & Rawls (2006).

    Equations (as reproduced e.g. in Alaya et al. 2017, Eq. 2–3):
    - theta1500t polynomial in S,C,OM (fractions in % units)
    - theta1500 = theta1500t + 0.14*theta1500t - 0.02
    """
    s = sand_pct / 100.0
    c = clay_pct / 100.0
    om = om_pct / 100.0
    theta1500t = (
        -0.024 * s
        + 0.487 * c
        + 0.006 * om
        + 0.005 * s * om
        - 0.013 * c * om
        + 0.068 * s * c
        + 0.031
    )
    return theta1500t + 0.14 * theta1500t - 0.02


def _saxton_rawls_theta33(sand_pct: np.ndarray, clay_pct: np.ndarray, om_pct: np.ndarray) -> np.ndarray:
    """
    Volumetric water content at 33 kPa (field capacity), Saxton & Rawls (2006).

    Equations (as reproduced e.g. in Alaya et al. 2017, Eq. 5–6).
    """
    s = sand_pct / 100.0
    c = clay_pct / 100.0
    om = om_pct / 100.0
    theta33t = (
        -0.251 * s
        + 0.195 * c
        + 0.011 * om
        + 0.006 * s * om
        - 0.027 * c * om
        + 0.452 * s * c
        + 0.299
    )
    return theta33t + 1.283 * theta33t**2 - 0.374 * theta33t - 0.015


def saxton_rawls_available_water_capacity_mm(
    *,
    sand_pct: np.ndarray,
    clay_pct: np.ndarray,
    soc_gkg: np.ndarray,
    depth_cm: float,
) -> np.ndarray:
    """
    Compute AWC (mm) over a soil depth using Saxton–Rawls (2006) PTF.

    Parameters
    ----------
    sand_pct, clay_pct:
        Particle size fractions in percent.
    soc_gkg:
        Soil organic carbon in g/kg (SoilGrids convention). Converted to OM%.
    depth_cm:
        Total depth over which to compute available water (cm).
    """
    # SOC g/kg → SOC % → OM % (van Bemmelen factor)
    soc_pct = np.asarray(soc_gkg, dtype=np.float32) / 10.0
    om_pct = soc_pct * 1.724
    theta33 = _saxton_rawls_theta33(np.asarray(sand_pct), np.asarray(clay_pct), om_pct)
    theta1500 = _saxton_rawls_theta1500(np.asarray(sand_pct), np.asarray(clay_pct), om_pct)
    awc_frac = np.clip(theta33 - theta1500, 0.01, 0.35)
    return awc_frac * (depth_cm * 10.0)


@dataclass
class SoilIngest:
    """
    Soil property cube builder from SoilGrids 2.0 with optional iSDA Africa gap fill.

    Produces an ``xr.Dataset`` with variables stacked into dims:
    - latitude, longitude: grid
    - property: soil variable name
    - depth: depth interval label (e.g. "0-5", "5-15", ...)
    """

    aoi: ee.Geometry
    properties: Iterable[str] | None = None
    depths_cm: Iterable[tuple[int, int]] | None = None
    scale: int = DEFAULT_SCALE_M
    chunks: dict[str, int] | None = None
    project: str | None = None
    use_isda_gapfill: bool = True

    def __post_init__(self) -> None:
        self.properties = tuple(self.properties) if self.properties is not None else DEFAULT_PROPERTIES
        self.depths_cm = tuple(self.depths_cm) if self.depths_cm is not None else DEFAULT_DEPTHS_CM
        self.chunks = self.chunks or {"latitude": 256, "longitude": 256}
        self._dataset: xr.Dataset | None = None

    def build(self) -> xr.Dataset:
        initialize_earth_engine(project=self.project)

        bands: list[ee.Image] = []
        band_names: list[str] = []
        for prop in self.properties:
            base_asset = SOILGRIDS_20_ASSET_TEMPLATE.format(property=prop)
            base_img = ee.Image(base_asset)
            for depth in self.depths_cm:
                band = _soilgrids_band_for_depth(depth)
                img = base_img.select(band)
                if self.use_isda_gapfill:
                    try:
                        isda = ee.Image(_isda_asset(prop, depth))
                        img = img.unmask(isda)
                    except Exception:
                        # iSDA not available for this property/depth or outside Africa.
                        pass
                name = f"{prop}__{_depth_label(depth)}"
                bands.append(img.rename(name))
                band_names.append(name)

        stacked = ee.Image.cat(bands).clip(self.aoi)
        ds = xr.open_dataset(
            stacked,
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

        # Convert the multi-var dataset to a single cube variable for easy joins.
        prop_names = [name.split("__", 1)[0] for name in band_names]
        depth_names = [name.split("__", 1)[1] for name in band_names]
        unique_props = list(dict.fromkeys(prop_names))
        unique_depths = list(dict.fromkeys(depth_names))

        cube = []
        for prop in unique_props:
            depth_slices = []
            for depth in unique_depths:
                key = f"{prop}__{depth}"
                if key not in ds.data_vars:
                    # Shouldn't happen, but keep the cube rectangular.
                    depth_slices.append(xr.full_like(next(iter(ds.data_vars.values())), np.nan))
                else:
                    depth_slices.append(ds[key])
            da = xr.concat(depth_slices, dim=xr.IndexVariable("depth", unique_depths))
            cube.append(da)
        data = xr.concat(cube, dim=xr.IndexVariable("property", unique_props))

        out = xr.Dataset({"soil": data})
        out.attrs.update(
            {
                "source": "Google Earth Engine",
                "collection": "SoilGrids2.0 (+iSDA Africa gap fill)",
                "scale_m": self.scale,
            }
        )
        self._dataset = out
        return out

    def available_water_capacity(self, *, depth_cm: int = 100) -> xr.DataArray:
        """
        AWC (mm) using Saxton–Rawls PTF from sand/clay/SOC.

        Uses the shallowest available layer values as proxies for the full profile when
        deeper layers are absent in the cube.
        """
        ds = self._dataset or self.build()
        soil = ds["soil"]

        def _pick(prop: str, depth: str) -> xr.DataArray:
            return soil.sel(property=prop, depth=depth)

        # Prefer 0-5 and 5-15 layers for texture and SOC; if missing, fall back to first.
        depth0 = "0-5" if "0-5" in soil["depth"].values else str(soil["depth"].values[0])
        sand = _pick("sand", depth0)
        clay = _pick("clay", depth0)
        soc = _pick("soc", depth0)

        awc = xr.apply_ufunc(
            saxton_rawls_available_water_capacity_mm,
            sand,
            clay,
            soc,
            kwargs={"depth_cm": float(depth_cm)},
            input_core_dims=[[], [], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
        ).rename("awc_mm")

        awc.attrs.update({"units": "mm", "method": "Saxton-Rawls-2006"})
        return awc

    def to_zarr(
        self,
        path: str | Path,
        *,
        mode: str = "w",
        include_awc: bool = True,
        awc_depth_cm: int = 100,
        consolidated: bool = True,
    ) -> Path:
        """
        Persist the soil cube to Zarr, optionally adding an ``awc_mm`` variable.

        This is the recommended output for :class:`~data.feature_store.FeatureStore`.
        """
        ds = self._dataset or self.build()
        out = ds
        if include_awc:
            out = out.copy()
            out["awc_mm"] = self.available_water_capacity(depth_cm=awc_depth_cm)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_zarr(out_path, mode=mode, consolidated=consolidated)
        return out_path


__all__ = [
    "SoilIngest",
    "DEFAULT_PROPERTIES",
    "DEFAULT_DEPTHS_CM",
    "saxton_rawls_available_water_capacity_mm",
]

