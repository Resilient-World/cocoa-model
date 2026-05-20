"""
ERA5 / CHIRPS daily climate ingest via Google Earth Engine → lazy xarray → Zarr.

Server-side band algebra in Earth Engine; materialization uses the Xarray ``ee``
backend (Xee). No raw NetCDF downloads.
"""

from __future__ import annotations

import math
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import xarray as xr

# Registers the ``ee`` Xarray backend (Xee).
import xee  # noqa: F401

from data.gee_auth import initialize_earth_engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ERA5_LAND_DAILY = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_DAILY = "ECMWF/ERA5/DAILY"
CHIRPS_DAILY = "UCSB-CHG/CHIRPS/DAILY"

KELVIN_OFFSET = 273.15

# Magnus (Alduchov & Eskridge 1996) — saturation vapor pressure [kPa]
MAGNUS_A = 0.61094
MAGNUS_B = 17.625
MAGNUS_C = 243.04

# FAO-56 reference grass
FAO_ALBEDO = 0.23
FAO_RS = 70.0  # s/m
FAO_GAMMA = 0.067  # kPa/°C (psychrometric constant, sea-level approx.)
WIND10_TO_WIND2_FACTOR = 4.87 / np.log(67.8 * 10.0 - 5.42)

# Root-zone depth weights (m) for ERA5-Land layers 1–3 (0–7, 7–28, 28–100 cm)
SM_LAYER_DEPTHS = (0.07, 0.21, 0.72)

DEFAULT_SCALE_M = 11_000  # ~0.1° ERA5-Land

OUTPUT_VARS = (
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


def _saturation_vapor_pressure_kpa(tmean_c: float) -> float:
    """Magnus saturation vapor pressure (kPa) at ``tmean_c`` (°C)."""
    return MAGNUS_A * math.exp(MAGNUS_B * tmean_c / (MAGNUS_C + tmean_c))


def _vpd_kpa(tmean_c: float, rh_pct: float) -> float:
    """Vapor pressure deficit (kPa) from mean temperature and RH (%)."""
    rh = max(0.0, min(100.0, rh_pct))
    es = _saturation_vapor_pressure_kpa(tmean_c)
    return es * (1.0 - rh / 100.0)


def _magnus_es_kpa(temp_c: ee.Image) -> ee.Image:
    """Saturation vapor pressure (kPa) from temperature (°C)."""
    return (
        ee.Image(MAGNUS_A)
        .multiply(temp_c.multiply(MAGNUS_B).divide(temp_c.add(MAGNUS_C)).exp())
    )


def _kelvin_to_celsius(img: ee.Image, band: str, new_name: str) -> ee.Image:
    return img.select(band).subtract(KELVIN_OFFSET).rename(new_name)


def _fao_et0_daily(
    tmean_c: ee.Image,
    rh_pct: ee.Image,
    wind10m: ee.Image,
    srad_mj: ee.Image,
) -> ee.Image:
    """
    FAO-56 Penman–Monteith reference ET0 (mm/day), grass reference.

    Rn ≈ (1 - albedo) * Rs with Rnl omitted (Rs-only simplification when only
    downward solar is available). G = 0.
    """
    es = _magnus_es_kpa(tmean_c)
    ea = es.multiply(rh_pct.divide(100.0))
    vpd = es.subtract(ea).max(0)

    delta = (
        es.multiply(MAGNUS_B)
        .multiply(MAGNUS_C)
        .divide(tmean_c.add(MAGNUS_C).pow(2))
    )

    u2 = wind10m.multiply(WIND10_TO_WIND2_FACTOR)
    rn = srad_mj.multiply(1.0 - FAO_ALBEDO)

    t_k = tmean_c.add(KELVIN_OFFSET)
    num_rad = delta.multiply(rn).multiply(0.408)
    num_aero = (
        ee.Image(FAO_GAMMA)
        .multiply(ee.Image(900).divide(t_k))
        .multiply(u2)
        .multiply(vpd)
    )
    den = delta.add(ee.Image(FAO_GAMMA).multiply(ee.Image(1).add(u2.multiply(0.34))))

    et0 = num_rad.add(num_aero).divide(den).max(0).rename("et0")
    return et0


def _build_daily_collection(
    aoi: ee.Geometry,
    start: str,
    end: str,
    *,
    chirps_for_precip: bool,
) -> ee.ImageCollection:
    """Assemble a daily ImageCollection with all target bands (server-side)."""
    era5_land = (
        ee.ImageCollection(ERA5_LAND_DAILY)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(
            [
                "temperature_2m",
                "dewpoint_temperature_2m",
                "u_component_of_wind_10m",
                "v_component_of_wind_10m",
                "surface_solar_radiation_downwards_sum",
                "volumetric_soil_water_layer_1",
                "volumetric_soil_water_layer_2",
                "volumetric_soil_water_layer_3",
                "total_precipitation_sum",
            ]
        )
    )
    era5_daily = (
        ee.ImageCollection(ERA5_DAILY)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(["maximum_2m_air_temperature", "minimum_2m_air_temperature"])
    )
    chirps = None
    if chirps_for_precip:
        chirps = (
            ee.ImageCollection(CHIRPS_DAILY)
            .filterDate(start, end)
            .filterBounds(aoi)
            .select(["precipitation"])
        )

    def _enrich(land_img: ee.Image) -> ee.Image:
        millis = land_img.date().millis()
        era5_img = era5_daily.filter(ee.Filter.eq("system:time_start", millis)).first()
        tmax = _kelvin_to_celsius(era5_img, "maximum_2m_air_temperature", "tmax")
        tmin = _kelvin_to_celsius(era5_img, "minimum_2m_air_temperature", "tmin")
        tmean = _kelvin_to_celsius(land_img, "temperature_2m", "tmean")

        t_dew = _kelvin_to_celsius(land_img, "dewpoint_temperature_2m", "t_dew")
        es_mean = _magnus_es_kpa(tmean)
        es_dew = _magnus_es_kpa(t_dew)
        rh = es_dew.divide(es_mean).multiply(100).clamp(0, 100).rename("rh_mean")
        vpd = es_mean.multiply(ee.Image(1).subtract(rh.divide(100))).rename("vpd_mean")

        u = land_img.select("u_component_of_wind_10m")
        v = land_img.select("v_component_of_wind_10m")
        wind10m = u.hypot(v).rename("wind10m")

        srad = (
            land_img.select("surface_solar_radiation_downwards_sum")
            .divide(1e6)
            .rename("srad")
        )

        sw1 = land_img.select("volumetric_soil_water_layer_1")
        sw2 = land_img.select("volumetric_soil_water_layer_2")
        sw3 = land_img.select("volumetric_soil_water_layer_3")
        sm_root = (
            sw1.multiply(SM_LAYER_DEPTHS[0])
            .add(sw2.multiply(SM_LAYER_DEPTHS[1]))
            .add(sw3.multiply(SM_LAYER_DEPTHS[2]))
            .divide(sum(SM_LAYER_DEPTHS))
            .rename("sm_root")
        )

        era5_precip_mm = land_img.select("total_precipitation_sum").multiply(1000)
        if chirps is not None:
            chirps_img = chirps.filter(ee.Filter.eq("system:time_start", millis)).first()
            chirps_mm = chirps_img.select("precipitation")
            precip = chirps_mm.unmask(era5_precip_mm).rename("precip")
        else:
            precip = era5_precip_mm.rename("precip")

        et0 = _fao_et0_daily(tmean, rh, wind10m, srad)
        cwd = et0.subtract(precip).rename("cwd")

        daily = ee.Image.cat(
            [tmax, tmin, tmean, rh, vpd, precip, et0, cwd, sm_root, wind10m, srad]
        ).copyProperties(land_img, ["system:time_start"])

        return daily.clip(aoi)

    return era5_land.map(_enrich)


class ERA5Ingest:
    """
    Ingest daily agrometeorology for an AOI and date range via Earth Engine + Xee.

    Results are lazy until computed; use :meth:`to_zarr` for chunked persistence.
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        start: str,
        end: str,
        *,
        chirps_for_precip: bool = True,
        chunks: dict[str, int] | None = None,
        scale: int = DEFAULT_SCALE_M,
        project: str | None = None,
    ) -> None:
        self.aoi = aoi
        self.start = start
        self.end = end
        self.chirps_for_precip = chirps_for_precip
        self.chunks = chunks or {"time": 30, "latitude": 256, "longitude": 256}
        self.scale = scale
        self.project = project
        self._dataset: xr.Dataset | None = None

    def build(self) -> xr.Dataset:
        """Open a lazy daily ``xarray.Dataset`` backed by Earth Engine."""
        initialize_earth_engine(project=self.project)

        ic = _build_daily_collection(
            self.aoi,
            self.start,
            self.end,
            chirps_for_precip=self.chirps_for_precip,
        )

        ds = xr.open_dataset(
            ic,
            engine="ee",
            geometry=self.aoi,
            scale=self.scale,
            chunks=self.chunks,
        )

        # Standardize dimension names and variable set
        rename_map: dict[str, str] = {}
        if "lat" in ds.dims:
            rename_map["lat"] = "latitude"
        if "lon" in ds.dims:
            rename_map["lon"] = "longitude"
        if rename_map:
            ds = ds.rename(rename_map)

        keep = [v for v in OUTPUT_VARS if v != "cwd_cum" and v in ds.data_vars]
        ds = ds[keep]

        ds["cwd_cum"] = ds["cwd"].cumsum(dim="time")

        ds.attrs.update(
            {
                "source": "Google Earth Engine",
                "era5_land_collection": ERA5_LAND_DAILY,
                "era5_collection": ERA5_DAILY,
                "chirps_collection": CHIRPS_DAILY if self.chirps_for_precip else "disabled",
                "start_date": self.start,
                "end_date": self.end,
                "magnus": "Alduchov & Eskridge 1996",
                "et0_method": "FAO-56 Penman-Monteith (grass reference)",
            }
        )

        self._dataset = ds
        return ds

    def to_zarr(self, path: str, mode: str = "w") -> None:
        """Materialize :meth:`build` output to a chunked Zarr store."""
        ds = self._dataset if self._dataset is not None else self.build()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(str(out), mode=mode)


def compute_derived_features(ds: xr.Dataset) -> xr.Dataset:
    """
    Add cocoa-oriented derived features on a daily climate dataset.

    Features
    --------
    gdd_cocoa:
        Growing degree-days (base 18 °C, cap 32 °C; Schwendenmann et al.).
    heat_days_above_32c:
        Binary indicator ``tmax > 32 °C``.
    dry_spell_max:
        Longest run of consecutive days with ``precip < 1`` mm.
    rolling means:
        30- and 90-day rolling means of ``vpd_mean``, ``cwd``, ``sm_root``.
    """
    out = ds.copy()

    tmax = out["tmax"]
    tmean = out["tmean"]
    precip = out["precip"]

    out["gdd_cocoa"] = (tmean.clip(min=18, max=32) - 18).clip(min=0)

    out["heat_days_above_32c"] = (tmax > 32.0).astype(np.int8)

    def _max_dry_spell(p: np.ndarray) -> float:
        mask = p < 1.0
        if not mask.any():
            return 0.0
        max_run = cur = 0
        for val in mask:
            if val:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0
        return float(max_run)

    precip_np = precip.values
    if precip_np.ndim == 3:
        spell = xr.apply_ufunc(
            _max_dry_spell,
            precip,
            input_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64],
        )
        out["dry_spell_max"] = spell
    else:
        out["dry_spell_max"] = xr.DataArray(_max_dry_spell(precip_np.ravel()))

    for window in (30, 90):
        out[f"vpd_mean_{window}d"] = out["vpd_mean"].rolling(time=window, min_periods=1).mean()
        out[f"cwd_{window}d"] = out["cwd"].rolling(time=window, min_periods=1).mean()
        out[f"sm_root_{window}d"] = out["sm_root"].rolling(time=window, min_periods=1).mean()

    return out


def _geometry_from_geojson(path: Path) -> ee.Geometry:
    """Load AOI polygon from GeoJSON via geopandas."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Empty GeoJSON: {path}")
    geom = gdf.geometry.unary_union
    return ee.Geometry(geom.__geo_interface__)


def main(argv: list[str] | None = None) -> int:
    """CLI: ingest ERA5-Land daily stack for an AOI to Zarr."""
    import argparse
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ERA5-Land daily ingest → Zarr")
    parser.add_argument("--aoi", type=Path, required=True, help="AOI GeoJSON path")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--out", type=Path, required=True, help="Output Zarr directory")
    parser.add_argument("--project", default=None, help="Earth Engine GCP project")
    args = parser.parse_args(argv)

    try:
        aoi = _geometry_from_geojson(args.aoi)
        ingest = ERA5Ingest(aoi, args.start, args.end, project=args.project)
        ingest.to_zarr(str(args.out))
        logging.getLogger(__name__).info("Wrote ERA5 Zarr to %s", args.out)
        return 0
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
