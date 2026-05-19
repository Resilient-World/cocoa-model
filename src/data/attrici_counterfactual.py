"""
Counterfactual ERA5-Land climate via ISIMIP3a GSWP3-W5E5 obsclim / counterclim deltas.

Uses the ISIMIP3a counterfactual methodology (Mengel et al. 2021, Geosci. Model Dev.,
14, 5269–5289; https://doi.org/10.5194/gmd-14-5269-2021) and GSWP3-W5E5 daily fields
(DOI 10.48364/ISIMIP.982724.3; CC0). ATTRICI v1.1 produced the counterclim experiment;
this module applies pre-computed ISIMIP deltas to an ERA5-Land factual stack — it does
**not** import the ATTRICI Python package (GPL boundary).

Delta adjustment (per variable, after regridding ISIMIP 0.5° → ERA5 grid)::

    cf = factual - (obsclim - counterclim)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd
import requests
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

logger = logging.getLogger(__name__)

MENGEL_2021_DOI = "10.5194/gmd-14-5269-2021"
ISIMIP_DATASET_DOI = "10.48364/ISIMIP.982724.3"

ISIMIP_FILES_BASE = (
    "https://files.isimip.org/ISIMIP3a/InputData/climate/atmosphere"
)
DEFAULT_CACHE_DIR = Path("data/external/isimip3a")
DEFAULT_ISIMIP_VARIABLES: tuple[str, ...] = (
    "tasmax",
    "tasmin",
    "tas",
    "pr",
    "hurs",
    "rsds",
    "sfcwind",
)
EXPERIMENTS: tuple[str, ...] = ("obsclim", "counterclim")

ISIMIP_TO_ERA5: dict[str, str] = {
    "tasmax": "tmax",
    "tasmin": "tmin",
    "tas": "tmean",
    "pr": "precip",
    "hurs": "rh_mean",
    "rsds": "srad",
    "sfcwind": "wind10m",
}

_TEMPERATURE_ISIMIP = frozenset({"tas", "tasmin", "tasmax"})
_KELVIN_THRESHOLD = 150.0
_OPEN_CHUNKS: dict[str, int] = {"time": 365}


def _isimip_file_url(experiment: str, variable: str, year: int) -> str:
    return (
        f"{ISIMIP_FILES_BASE}/{experiment}/global/daily/historical/GSWP3-W5E5/"
        f"gswp3-w5e5_{experiment}_{variable}_global_daily_{year}_{year}.nc"
    )


def _cache_file_path(cache_dir: Path, experiment: str, variable: str, year: int) -> Path:
    name = f"gswp3-w5e5_{experiment}_{variable}_global_daily_{year}_{year}.nc"
    return cache_dir / experiment / variable / name


def _meta_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".meta.json")


def _download_cached(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` if missing or remote ETag/size changed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_file = _meta_path(dest)
    prior: dict[str, Any] = {}
    if meta_file.is_file():
        prior = json.loads(meta_file.read_text(encoding="utf-8"))

    headers: dict[str, str] = {}
    if dest.is_file() and prior.get("etag"):
        headers["If-None-Match"] = str(prior["etag"])

    head = requests.head(url, timeout=timeout, allow_redirects=True)
    head.raise_for_status()
    remote_etag = head.headers.get("ETag")
    remote_size = head.headers.get("Content-Length")

    if dest.is_file():
        if remote_etag and prior.get("etag") == remote_etag:
            return dest
        if remote_size and dest.stat().st_size == int(remote_size):
            return dest

    logger.info("Downloading %s -> %s", url, dest)
    etag_out = remote_etag
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
        resp.raise_for_status()
        if resp.status_code == 304 and dest.is_file():
            return dest
        etag_out = resp.headers.get("ETag", remote_etag)
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)

    meta = {
        "url": url,
        "etag": etag_out,
        "size": dest.stat().st_size,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def _parse_year(date_str: str) -> int:
    return int(pd.Timestamp(date_str).year)


def _coord_names(ds: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    if "latitude" in ds.coords or "latitude" in getattr(ds, "dims", {}):
        return "latitude", "longitude"
    return "lat", "lon"


def _subset_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    west, south, east, north = bbox
    lat_name, lon_name = _coord_names(ds)
    out = ds.sel({lat_name: slice(south, north)})

    lon = out[lon_name]
    if float(lon.max()) > 180.0 and west < 0:
        west_360 = west % 360.0
        east_360 = east % 360.0
        if west_360 <= east_360:
            out = out.sel({lon_name: slice(west_360, east_360)})
        else:
            part_a = out.sel({lon_name: slice(west_360, 360.0)})
            part_b = out.sel({lon_name: slice(0.0, east_360)})
            out = xr.concat([part_a, part_b], dim=lon_name)
    else:
        out = out.sel({lon_name: slice(west, east)})
    return out


def _normalize_isimip_units(da: xr.DataArray, isimip_var: str) -> xr.DataArray:
    """Kelvin → °C for temperature; kg m⁻² s⁻¹ → mm d⁻¹ for precipitation."""
    out = da
    if isimip_var in _TEMPERATURE_ISIMIP:
        sample = float(out.isel({d: 0 for d in out.dims if d != "time"}, drop=True).mean().values)
        if sample > _KELVIN_THRESHOLD:
            out = out - KELVIN_OFFSET
    if isimip_var == "pr":
        out = out * 86400.0
    return out


def _resolve_isimip_var(ds: xr.Dataset, isimip_var: str) -> xr.DataArray:
    if isimip_var in ds.data_vars:
        return ds[isimip_var]
    for name in ds.data_vars:
        if isimip_var in name.lower():
            return ds[name]
    raise KeyError(f"Variable {isimip_var!r} not in {list(ds.data_vars)}")


def _regrid_to_factual(
    source: xr.DataArray,
    factual: xr.Dataset,
    *,
    method: str = "bilinear",
) -> xr.DataArray:
    import xesmf as xe

    src_lat, src_lon = _coord_names(source)
    tgt_lat, tgt_lon = _coord_names(factual)
    src_grid = xr.Dataset({"lat": source[src_lat], "lon": source[src_lon]})
    dst_grid = xr.Dataset({"lat": factual[tgt_lat], "lon": factual[tgt_lon]})
    regridder = xe.Regridder(src_grid, dst_grid, method, reuse_weights=False)
    out = regridder(source.transpose(src_lat, src_lon, ...), keep_attrs=True)
    regridder.clean_weight_file()
    return out


def _recompute_vpd_et0_cwd(ds: xr.Dataset) -> xr.Dataset:
    """Recompute ``vpd_mean``, ``et0``, and ``cwd`` from adjusted daily drivers."""
    out = ds.copy()
    tmean = out["tmean"]
    rh = out["rh_mean"].clip(0, 100)
    es = MAGNUS_A * np.exp(MAGNUS_B * tmean / (MAGNUS_C + tmean))
    out["vpd_mean"] = (es * (1.0 - rh / 100.0)).clip(min=0)

    u2 = out["wind10m"] * WIND10_TO_WIND2_FACTOR
    rn = out["srad"] * (1.0 - FAO_ALBEDO)
    t_k = tmean + KELVIN_OFFSET
    delta_slope = es * MAGNUS_B * MAGNUS_C / (tmean + MAGNUS_C) ** 2
    vpd = out["vpd_mean"]
    num_rad = delta_slope * rn * 0.408
    num_aero = FAO_GAMMA * (900.0 / t_k) * u2 * vpd
    den = delta_slope + FAO_GAMMA * (1.0 + 0.34 * u2)
    out["et0"] = ((num_rad + num_aero) / den).clip(min=0)
    out["cwd"] = out["et0"] - out["precip"]
    return out


def _aoi_bounds(aoi: ee.Geometry | Any | None) -> tuple[float, float, float, float]:
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
        from data.gee_auth import initialize_earth_engine

        initialize_earth_engine()
        coords = aoi.bounds().getInfo()["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))

    if isinstance(aoi, (tuple, list)) and len(aoi) == 4:
        west, south, east, north = aoi
        return float(west), float(south), float(east), float(north)

    raise TypeError(f"Unsupported AOI type: {type(aoi)!r}")


class ISIMIPCounterfactualLoader:
    """
    Download (cached) and open ISIMIP3a GSWP3-W5E5 obsclim / counterclim for an AOI.

    Opens files lazily with ``chunks={"time": 365}`` and subsets lat/lon before
    concatenating years — never loads a full global field into RAM.
    """

    def __init__(
        self,
        aoi_bbox: tuple[float, float, float, float] | None = None,
        start: str = "",
        end: str = "",
        *,
        bbox: tuple[float, float, float, float] | None = None,
        variables: tuple[str, ...] = DEFAULT_ISIMIP_VARIABLES,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ) -> None:
        resolved = aoi_bbox if aoi_bbox is not None else bbox
        if resolved is None:
            raise TypeError("ISIMIPCounterfactualLoader requires aoi_bbox or bbox")
        self.aoi_bbox = resolved
        self.start = start
        self.end = end
        self.variables = variables
        self.cache_dir = Path(cache_dir)
        self._start_year = _parse_year(start)
        self._end_year = _parse_year(end)

    def _ensure_year_files(self, experiment: str, variable: str) -> list[Path]:
        paths: list[Path] = []
        for year in range(self._start_year, self._end_year + 1):
            dest = _cache_file_path(self.cache_dir, experiment, variable, year)
            url = _isimip_file_url(experiment, variable, year)
            paths.append(_download_cached(url, dest))
        return paths

    def _open_variable(self, experiment: str, variable: str) -> xr.DataArray:
        paths = self._ensure_year_files(experiment, variable)
        pieces: list[xr.DataArray] = []
        for path in paths:
            ds = xr.open_dataset(path, chunks=_OPEN_CHUNKS)
            ds = _subset_bbox(ds, self.aoi_bbox)
            da = _normalize_isimip_units(_resolve_isimip_var(ds, variable), variable)
            pieces.append(da.load())
            ds.close()
        combined = xr.concat(pieces, dim="time")
        combined.name = variable
        start_ts = pd.Timestamp(self.start)
        end_ts = pd.Timestamp(self.end)
        return combined.sel(time=slice(start_ts, end_ts))

    def load_experiment(self, experiment: str) -> xr.Dataset:
        """Load all configured variables for ``obsclim`` or ``counterclim``."""
        if experiment not in EXPERIMENTS:
            raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {experiment!r}")

        data_vars: dict[str, xr.DataArray] = {}
        for var in self.variables:
            logger.info("Loading ISIMIP %s %s (%d–%d)", experiment, var, self._start_year, self._end_year)
            data_vars[var] = self._open_variable(experiment, var)

        ds = xr.Dataset(data_vars)
        ds.attrs.update(
            {
                "experiment": experiment,
                "source": "ISIMIP3a GSWP3-W5E5",
                "mengel_2021_doi": MENGEL_2021_DOI,
                "isimip_doi": ISIMIP_DATASET_DOI,
            }
        )
        return ds

    def load_obsclim(self) -> xr.Dataset:
        return self.load_experiment("obsclim")

    def load_counterclim(self) -> xr.Dataset:
        return self.load_experiment("counterclim")

    def load(self, experiment: str) -> xr.Dataset:
        """Alias for :meth:`load_experiment`."""
        return self.load_experiment(experiment)


def _factual_variable_key(factual: xr.Dataset, isimip_var: str, era5_var: str) -> str | None:
    if era5_var in factual.data_vars:
        return era5_var
    if isimip_var in factual.data_vars:
        return isimip_var
    return None


def compute_counterfactual_delta(
    factual_era5: xr.Dataset,
    obsclim_isimip: xr.Dataset,
    counterclim_isimip: xr.Dataset,
    *,
    regrid_method: str = "bilinear",
    skip_regrid: bool = False,
    clip_precip: bool = False,
) -> xr.Dataset:
    """
    Apply ISIMIP obsclim–counterclim deltas on the ERA5-Land grid.

    For each mapped variable::

        cf = factual - (obsclim_regridded - counterclim_regridded)

    Then recomputes ``vpd_mean``, ``et0``, ``cwd``, ``cwd_cum``, copies ``sm_root``
    from factual where unchanged, and runs :func:`data.era5_ingest.compute_derived_features`.
    """
    adjusted: dict[str, xr.DataArray] = {}

    for isimip_var, era5_var in ISIMIP_TO_ERA5.items():
        factual_key = _factual_variable_key(factual_era5, isimip_var, era5_var)
        if factual_key is None:
            continue
        obs = _resolve_isimip_var(obsclim_isimip, isimip_var)
        counter = _resolve_isimip_var(counterclim_isimip, isimip_var)
        if skip_regrid:
            obs_r = obs
            counter_r = counter
        else:
            obs_r = _regrid_to_factual(obs, factual_era5, method=regrid_method)
            counter_r = _regrid_to_factual(counter, factual_era5, method=regrid_method)
        delta = obs_r - counter_r
        out_key = factual_key if skip_regrid else era5_var
        adjusted[out_key] = (factual_era5[factual_key] - delta).astype(np.float32)

    if not adjusted:
        raise ValueError("No ERA5 variables could be adjusted; check factual and ISIMIP inputs")

    cf = xr.Dataset(adjusted)
    for coord in factual_era5.coords:
        if coord not in cf.coords and coord in factual_era5.dims:
            cf = cf.assign_coords({coord: factual_era5[coord]})

    if clip_precip:
        for precip_name in ("precip", "pr"):
            if precip_name in cf:
                cf[precip_name] = cf[precip_name].clip(min=0)

    if skip_regrid:
        cf.attrs.update({"counterfactual": True, "method": "isimip_delta"})
        return cf

    required_for_derived = {"tmean", "rh_mean", "wind10m", "srad", "precip"}
    if required_for_derived.issubset(cf.data_vars):
        cf = _recompute_vpd_et0_cwd(cf)
        cf["cwd_cum"] = cf["cwd"].cumsum(dim="time")
        if "sm_root" in factual_era5.data_vars:
            cf["sm_root"] = factual_era5["sm_root"]
        cf = compute_derived_features(cf)

    cf.attrs.update(
        {
            "counterfactual": True,
            "method": "isimip_delta",
            "mengel_2021_doi": MENGEL_2021_DOI,
            "isimip_doi": ISIMIP_DATASET_DOI,
        }
    )
    return cf


def attrici_counterfactual_stack(
    aoi: ee.Geometry | Any,
    start: str,
    end: str,
    factual_ds: xr.Dataset,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    variables: tuple[str, ...] = DEFAULT_ISIMIP_VARIABLES,
) -> xr.Dataset:
    """
    End-to-end counterfactual ERA5-Land stack for an AOI and date range.

    Loads ISIMIP obsclim + counterclim, regrids, applies deltas, and returns a
    dataset with the same schema as :mod:`data.era5_ingest` factual output.
    """
    bbox = _aoi_bounds(aoi)
    loader = ISIMIPCounterfactualLoader(
        bbox,
        start,
        end,
        variables=variables,
        cache_dir=cache_dir,
    )
    obsclim = loader.load_obsclim()
    counterclim = loader.load_counterclim()
    return compute_counterfactual_delta(factual_ds, obsclim, counterclim)


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    return parts[0], parts[1], parts[2], parts[3]


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build counterfactual ERA5-Land Zarr from ISIMIP3a GSWP3-W5E5 deltas.",
    )
    parser.add_argument(
        "--aoi-bbox",
        type=_parse_bbox,
        required=True,
        help="AOI bounds as west,south,east,north (EPSG:4326)",
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--factual-zarr", type=Path, required=True, help="Factual ERA5 Zarr path")
    parser.add_argument(
        "--output-zarr",
        type=Path,
        required=True,
        help="Output counterfactual Zarr path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"ISIMIP download cache (default: {DEFAULT_CACHE_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    factual = xr.open_zarr(args.factual_zarr, chunks=_OPEN_CHUNKS)
    cf = attrici_counterfactual_stack(
        args.aoi_bbox,
        args.start,
        args.end,
        factual,
        cache_dir=args.cache_dir,
    )
    args.output_zarr.parent.mkdir(parents=True, exist_ok=True)
    cf.to_zarr(args.output_zarr, mode="w")
    logger.info("Wrote counterfactual stack to %s", args.output_zarr)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
