"""
ISIMIP3a counterclim ingest (GSWP3-W5E5, 0.5°) aligned to ERA5-Land grids.

Downloads Mengel et al. (2021) counterfactual climate (``climate_scenario=counterclim``)
via the ISIMIP file-list API and regrids onto grids from :mod:`data.era5_ingest`.

This module never imports the ``attrici`` package (GPL boundary). Running ATTRICI from
source is delegated to ``scripts/run_attrici_subprocess.py`` (Prompt 3).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd
import pooch
import requests
import xarray as xr

logger = logging.getLogger(__name__)

# Map ERA5-Land names (``era5_ingest``) → ISIMIP short names for delta attribution
_ERA5_TO_ISIMIP: dict[str, str] = {
    "tmean": "tas",
    "tmin": "tasmin",
    "tmax": "tasmax",
    "precip": "pr",
    "rh_mean": "hurs",
    "srad": "rsds",
}

ISIMIP_API_V1 = "https://data.isimip.org/api/v1"
ISIMIP_FILES_BASE = "https://files.isimip.org"
ZENODO_RECORD_ID = "5036364"
PR_WET_DAY_MM = 0.1  # Mengel et al. 2021 §3.2.3 (Bernoulli–gamma threshold)

DEFAULT_VARIABLES: tuple[str, ...] = ("tas", "tasmin", "tasmax", "pr", "hurs", "rsds")
_TAS_DERIVED: tuple[str, ...] = ("tasrange", "tasskew")
_ADDITIVE_DELTA_VARS: frozenset[str] = frozenset({"tas", "tasmin", "tasmax", "rsds", "hurs"})
_KELVIN_THRESHOLD = 150.0

_REGISTRY: pooch.Pooch | None = None


def _get_registry(cache_dir: Path) -> pooch.Pooch:
    global _REGISTRY
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _REGISTRY is None or Path(_REGISTRY.path) != cache_dir.resolve():
        _REGISTRY = pooch.create(
            path=str(cache_dir),
            base_url=ISIMIP_FILES_BASE + "/",
            registry={},
            urls={},
        )
    return _REGISTRY


def _expanded_variables(variables: Sequence[str]) -> list[str]:
    """Include tasrange/tasskew when tasmin/tasmax requested (ATTRICI §3.2.2)."""
    out: list[str] = []
    need_tas_derived = "tasmin" in variables or "tasmax" in variables
    for v in variables:
        if v not in out:
            out.append(v)
    if need_tas_derived:
        for v in _TAS_DERIVED:
            if v not in out:
                out.append(v)
        if "tas" not in out:
            out.insert(0, "tas")
    return out


def _aoi_bounds(
    aoi: ee.Geometry | Any | None,
) -> tuple[float, float, float, float]:
    """Return ``(west, south, east, north)`` in EPSG:4326."""
    if aoi is None:
        return -180.0, -90.0, 180.0, 90.0

    try:
        from shapely.geometry.base import BaseGeometry

        if isinstance(aoi, BaseGeometry):
            west, south, east, north = aoi.bounds
            return float(west), float(south), float(east), float(north)
    except ImportError:
        pass

    if isinstance(aoi, ee.Geometry):
        coords = aoi.bounds().getInfo()["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))

    raise TypeError(f"Unsupported AOI type: {type(aoi)!r}")


def _parse_year(date_str: str) -> int:
    return int(pd.Timestamp(date_str).year)


def _isimip_list_files(
    *,
    variables: Sequence[str],
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    """Query ISIMIP v1 ``/files/`` endpoint (paginated)."""
    params: dict[str, Any] = {
        "simulation_round": "ISIMIP3a",
        "climate_scenario": "counterclim",
        "climate_forcing": "gswp3-w5e5",
        "time_step": "daily",
        "limit": 100,
    }
    files: list[dict[str, Any]] = []
    url = f"{ISIMIP_API_V1}/files/"
    while url:
        response = requests.get(url, params=params if url.endswith("/files/") else None, timeout=120)
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("results", []):
            spec = item.get("specifiers") or {}
            var = spec.get("climate_variable")
            sy = int(spec.get("start_year", 0))
            ey = int(spec.get("end_year", 0))
            if var not in variables:
                continue
            if ey < start_year or sy > end_year:
                continue
            files.append(item)
        url = payload.get("next")
        params = None
    return files


def _zenodo_fallback_files(
    variables: Sequence[str],
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    """Best-effort Zenodo 5036364 file list when the ISIMIP API is unavailable."""
    api = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    response = requests.get(api, timeout=60)
    response.raise_for_status()
    files: list[dict[str, Any]] = []
    for entry in response.json().get("files", []):
        key = entry.get("key", "")
        if not key.endswith(".nc"):
            continue
        var = next((v for v in variables if f"_{v}_" in key or f"_{v}." in key), None)
        if var is None:
            continue
        years = [int(y) for y in key.replace(".nc", "").split("_") if y.isdigit() and len(y) == 4]
        if years and (max(years) < start_year or min(years) > end_year):
            continue
        files.append(
            {
                "name": key,
                "path": key,
                "checksum": entry.get("checksum"),
                "checksum_type": entry.get("checksum_type", "md5"),
                "download_url": entry["links"]["self"],
            }
        )
    if not files:
        raise FileNotFoundError(
            f"No Zenodo {ZENODO_RECORD_ID} files matched variables={variables!r} "
            f"years={start_year}-{end_year}"
        )
    return files


def _verify_sha512(path: Path, expected: str) -> None:
    digest = hashlib.sha512()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-512 mismatch for {path.name}")


def _download_isimip_file(meta: dict[str, Any], registry: pooch.Pooch) -> Path:
    path_key = meta["path"]
    fname = meta["name"]
    cache_name = f"{hashlib.sha256(path_key.encode()).hexdigest()[:16]}_{fname}"
    if cache_name in registry.registry:
        return Path(registry.fetch(cache_name))

    url = meta.get("download_url") or f"{ISIMIP_FILES_BASE}/{path_key}"
    dest = Path(registry.abspath) / cache_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s", fname)
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

    checksum = meta.get("checksum")
    if meta.get("checksum_type") == "sha512" and checksum:
        _verify_sha512(dest, checksum)

    registry.registry[cache_name] = fname
    registry.urls[cache_name] = url
    registry.dump(registry.abspath / "registry.txt")
    return dest


def _open_variable_timeseries(paths: list[Path], var: str) -> xr.DataArray:
    sorted_paths = sorted(paths)
    ds = xr.open_mfdataset(sorted_paths, combine="by_coords", parallel=False)
    try:
        da = ds[var] if var in ds else ds[list(ds.data_vars)[0]]
        return da.load()
    finally:
        ds.close()


def _harmonize_coords(ds: xr.Dataset) -> xr.Dataset:
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


def _subset_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    west, south, east, north = bbox
    return ds.sel(lat=slice(south, north), lon=slice(west, east))


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    if float(da.mean(skipna=True)) > _KELVIN_THRESHOLD:
        return da - 273.15
    return da


def reconstruct_tasmin_tasmax(ds: xr.Dataset) -> xr.Dataset:
    """Piani et al. 2010 / ATTRICI §3.2.2 from tas, tasrange, tasskew."""
    if not all(v in ds for v in ("tas", "tasrange", "tasskew")):
        return ds
    out = ds.copy()
    tas = _to_celsius(ds["tas"])
    tasrange = ds["tasrange"]
    tasskew = ds["tasskew"]
    out["tasmin"] = (tas - tasrange * tasskew).rename("tasmin")
    out["tasmax"] = (out["tasmin"] + tasrange).rename("tasmax")
    return out


def compute_attribution_deltas(
    factual: xr.Dataset,
    counterfactual: xr.Dataset,
    variables: Sequence[str],
) -> xr.Dataset:
    """
    Attribution deltas between factual and counterfactual fields.

    Additive for ``tas``, ``tasmin``, ``tasmax``, ``rsds``, ``hurs``. For ``pr``,
    returns ``log(factual / counterfactual)`` on wet days (≥ 0.1 mm/d) only
    (Mengel et al. 2021 §3.2.3).
    """
    deltas: dict[str, xr.DataArray] = {}
    for var in variables:
        if var not in factual or var not in counterfactual:
            logger.warning("Skipping delta for %s (missing in factual or counterfactual)", var)
            continue
        f = factual[var]
        c = counterfactual[var]
        if var == "pr":
            wet = (f >= PR_WET_DAY_MM) & (c >= PR_WET_DAY_MM)
            ratio = xr.where(wet, f / c.where(c > 0, np.nan), np.nan)
            deltas[f"{var}_delta"] = xr.where(wet, np.log(ratio), np.nan).rename(f"{var}_delta")
        elif var in _ADDITIVE_DELTA_VARS:
            deltas[f"{var}_delta"] = (f - c).rename(f"{var}_delta")
        else:
            deltas[f"{var}_delta"] = (f - c).rename(f"{var}_delta")
    return xr.Dataset(deltas)


def _lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    if "latitude" in ds.dims or "latitude" in ds.coords:
        return "latitude", "longitude"
    return "lat", "lon"


def _regrid_xesmf(
    source: xr.DataArray,
    target: xr.Dataset,
    method: str,
) -> xr.DataArray:
    import xesmf as xe

    lat_name, lon_name = _lat_lon_names(source.to_dataset(name=source.name))
    tgt_lat, tgt_lon = _lat_lon_names(target)

    src_grid = xr.Dataset(
        {
            "lat": source[lat_name],
            "lon": source[lon_name],
        }
    )
    dst_grid = xr.Dataset(
        {
            "lat": target[tgt_lat],
            "lon": target[tgt_lon],
        }
    )
    regridder = xe.Regridder(src_grid, dst_grid, method=method, reuse_weights=False)
    out = regridder(source.transpose(lat_name, lon_name, ...), keep_attrs=True)
    regridder.clean_weight_file()
    return out


def _regrid_interp(source: xr.DataArray, target: xr.Dataset) -> xr.DataArray:
    lat_name, lon_name = _lat_lon_names(source.to_dataset(name=source.name))
    tgt_lat, tgt_lon = _lat_lon_names(target)
    interp_method = "nearest" if source.name == "pr" else "linear"
    return source.interp(
        {lat_name: target[tgt_lat], lon_name: target[tgt_lon]},
        method=interp_method,
    )


class CounterfactualClimate:
    """
    Fetch ISIMIP3a counterclim (GSWP3-W5E5) and align to an ERA5-Land target grid.

    Parameters
    ----------
    aoi:
        Earth Engine or Shapely geometry defining the spatial subset.
    start, end:
        Date bounds (inclusive start, inclusive end) for the time dimension.
    variables:
        ISIMIP variable short names to load.
    cache_dir:
        Local cache for downloaded NetCDF files (SHA-512 verified when provided).
    target_grid:
        Factual ERA5-Land (or compatible) dataset used as the regrid target in :meth:`build`.
    """

    def __init__(
        self,
        aoi: ee.Geometry | Any | None,
        start: str,
        end: str,
        variables: Sequence[str] = DEFAULT_VARIABLES,
        cache_dir: str | Path = "data/counterclim",
        target_grid: xr.Dataset | None = None,
    ) -> None:
        self.aoi = aoi
        self.start = start
        self.end = end
        self.variables = tuple(variables)
        self.cache_dir = Path(cache_dir)
        self.target_grid = target_grid
        self._bbox = _aoi_bounds(aoi)
        self._start_year = _parse_year(start)
        self._end_year = _parse_year(end)
        self._dataset: xr.Dataset | None = None

    def fetch(self) -> xr.Dataset:
        """Download (or load cached) ISIMIP3a counterclim NetCDFs for the requested variables and time slice."""
        registry = _get_registry(self.cache_dir)
        query_vars = _expanded_variables(self.variables)

        try:
            file_metas = _isimip_list_files(
                variables=query_vars,
                start_year=self._start_year,
                end_year=self._end_year,
            )
        except requests.RequestException as exc:
            logger.warning("ISIMIP API failed (%s); trying Zenodo %s", exc, ZENODO_RECORD_ID)
            file_metas = _zenodo_fallback_files(query_vars, self._start_year, self._end_year)

        if not file_metas:
            raise FileNotFoundError(
                "No ISIMIP3a counterclim files matched the query. "
                "Check variables, dates, and network access."
            )

        by_var: dict[str, list[Path]] = {v: [] for v in query_vars}
        for meta in file_metas:
            var = (meta.get("specifiers") or {}).get("climate_variable")
            if var is None and "name" in meta:
                var = next((v for v in query_vars if v in meta["name"]), None)
            if var is None:
                continue
            by_var[var].append(_download_isimip_file(meta, registry))

        pieces: dict[str, xr.DataArray] = {}
        for var, paths in by_var.items():
            if not paths:
                logger.warning("No files downloaded for variable %s", var)
                continue
            pieces[var] = _open_variable_timeseries(paths, var)

        if not pieces:
            raise FileNotFoundError("No counterclim variables could be loaded")

        ds = _harmonize_coords(xr.Dataset(pieces))
        ds = _subset_bbox(ds, self._bbox)
        ds = ds.sel(time=slice(self.start, self.end))
        ds = reconstruct_tasmin_tasmax(ds)
        self._dataset = ds
        return ds

    def regrid_to(self, target: xr.Dataset, method: str = "bilinear") -> xr.Dataset:
        """
        Regrid counterfactual fields onto ``target``'s horizontal grid.

        Uses xESMF (conservative for ``pr``, bilinear otherwise) when available;
        falls back to :py:meth:`xarray.DataArray.interp`.
        """
        if self._dataset is None:
            raise RuntimeError("Call fetch() before regrid_to()")

        regridded: dict[str, xr.DataArray] = {}
        use_xesmf = True
        try:
            import xesmf  # noqa: F401
        except ImportError:
            use_xesmf = False
            logger.warning("xesmf not installed; falling back to xarray.interp for regridding")

        for var in self._dataset.data_vars:
            da = self._dataset[var]
            regrid_method = "conservative" if var == "pr" else method
            if use_xesmf:
                try:
                    regridded[var] = _regrid_xesmf(da, target, regrid_method)
                    continue
                except Exception as exc:
                    logger.warning("xesmf failed for %s (%s); using interp", var, exc)
            regridded[var] = _regrid_interp(da, target)

        out = xr.Dataset(regridded)
        if "time" in self._dataset.coords:
            out = out.assign_coords(time=self._dataset.time)
            out = out.sel(time=slice(self.start, self.end), method="nearest")
        return out

    def build(self) -> xr.Dataset:
        """
        ``fetch`` → AOI subset → regrid to ``target_grid`` → factual–counterfactual deltas.

        Deltas use ERA5 variable names from ``target_grid`` mapped through
        :data:`counterfactual.delta_downscaler.ISIMIP_TO_ERA5`.
        """
        if self.target_grid is None:
            raise ValueError("target_grid is required for build()")

        counter = self.fetch()
        counter_on_target = self.regrid_to(self.target_grid)

        factual_parts: dict[str, xr.DataArray] = {}
        for era5_var, isimip_var in _ERA5_TO_ISIMIP.items():
            if isimip_var not in self.variables:
                continue
            if era5_var in self.target_grid:
                factual_parts[isimip_var] = self.target_grid[era5_var]
        if not factual_parts:
            self._dataset = counter_on_target
            return counter_on_target

        factual_isimip = xr.Dataset(factual_parts)
        deltas = compute_attribution_deltas(
            factual_isimip,
            counter_on_target,
            [v for v in self.variables if v in factual_parts],
        )
        merged = xr.merge([counter_on_target, deltas], compat="override")
        self._dataset = merged
        return merged

    def to_zarr(self, path: str, mode: str = "w") -> None:
        """Persist :meth:`build` / :meth:`fetch` output to Zarr."""
        if self._dataset is None:
            self.build() if self.target_grid is not None else self.fetch()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._dataset.to_zarr(str(out), mode=mode)
