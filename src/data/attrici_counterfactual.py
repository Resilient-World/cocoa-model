"""
ATTRICI v2.0.1 counterfactual climate via subprocess (GPLv3 boundary).

Never ``import attrici`` in this module — all detrending is delegated to the
``attrici`` CLI (PyMC5/Scipy backend per v2.0.1). See ``NOTICE.md`` and
``docs/licensing/ATTRICI_GPL_BOUNDARY.md``.

Methodology: Mengel et al. (2021), *Geosci. Model Dev.* 14, 5269–5284.
"""

from __future__ import annotations

import structlog

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from counterfactual.attrici_runner import (
    ATTRICIRunner,
    SUPPORTED_VARIABLES,
    load_counterfactual,
)
from data.attrici_fast_detrend import recompute_derived_counterfactuals

log = structlog.get_logger(__name__)

MENGEL_2021_REF = "Mengel et al. (2021), Geosci. Model Dev. 14, 5269–5284 (ATTRICI)"

# CASEJ / ALMANAC / yield-surrogate ERA5 names ↔ ISIMIP short names (ATTRICI v2 CLI)
ERA5_VARIABLES: tuple[str, ...] = (
    "tmax",
    "tmin",
    "tmean",
    "precip",
    "rh_mean",
    "srad",
    "wind10m",
)
ISIMIP_ALIASES: dict[str, str] = {
    "tas": "tmean",
    "tasmax": "tmax",
    "tasmin": "tmin",
    "pr": "precip",
    "hurs": "rh_mean",
    "rsds": "srad",
    "sfcwind": "wind10m",
}


@dataclass(frozen=True)
class RegionBounds:
    """Geographic subset for counterfactual generation."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def as_dict(self) -> dict[str, float]:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


@dataclass(frozen=True)
class TimeRange:
    """Inclusive calendar-year window."""

    start_year: int
    end_year: int

    def as_dict(self) -> dict[str, int]:
        return {"start_year": self.start_year, "end_year": self.end_year}


def normalize_variables(variables: Sequence[str]) -> tuple[str, ...]:
    """Map ISIMIP short names to ERA5 ingest names."""
    out: list[str] = []
    for v in variables:
        key = v.strip().lower()
        era5 = ISIMIP_ALIASES.get(key, key)
        if era5 not in SUPPORTED_VARIABLES:
            log.warning("Variable %s not in ATTRICI runner support; skipping", v)
            continue
        if era5 == "tmean":
            # Runner detrends tas via tmean if present; prefer tmax+tmin for derived recompute
            if "tmax" not in out:
                out.append("tmax")
            if "tmin" not in out:
                out.append("tmin")
            continue
        if era5 not in out:
            out.append(era5)
    return tuple(out)


class ATTRICICounterfactual:
    """
    Build/cache counterfactual ERA5 Zarr from factual ERA5-Land via ATTRICI v2 CLI.

    Parameters
    ----------
    factual_zarr:
        Path to factual daily ERA5-Land (or compatible) Zarr store.
    gmt_file:
        SSA-smoothed global-mean temperature NetCDF for ATTRICI ``detrend``.
    cache_dir:
        Directory for content-addressed counterfactual Zarr caches.
    attrici_bin:
        ATTRICI executable (default ``attrici`` on ``PATH``, or ``.venv-attrici/bin/attrici``).
    backend:
        ATTRICI solver: ``scipy`` (fast) or ``pymc5``.
    n_workers:
        Reserved for grid-parallel extensions; passed where supported.
    """

    def __init__(
        self,
        factual_zarr: Path | str,
        *,
        gmt_file: Path | str,
        cache_dir: Path | str | None = None,
        attrici_bin: str | None = None,
        backend: str = "scipy",
        n_workers: int = 4,
    ) -> None:
        self.factual_zarr = Path(factual_zarr)
        self.gmt_file = Path(gmt_file)
        self.cache_dir = Path(cache_dir or "data/cache/attrici_counterfactual")
        self.attrici_bin = attrici_bin or "attrici"
        self.backend = backend
        self.n_workers = n_workers

    def cache_key(
        self,
        variables: Sequence[str],
        *,
        region: RegionBounds | None = None,
        time_range: TimeRange | None = None,
    ) -> str:
        """Stable hash over inputs (variable set, region, time, factual path mtime)."""
        payload: dict[str, Any] = {
            "factual": str(self.factual_zarr.resolve()),
            "variables": sorted(normalize_variables(variables)),
            "gmt": str(self.gmt_file.resolve()),
            "backend": self.backend,
            "attrici_version": "v2.0.1",
        }
        if self.factual_zarr.exists():
            payload["factual_mtime"] = self.factual_zarr.stat().st_mtime
        if region is not None:
            payload["region"] = region.as_dict()
        if time_range is not None:
            payload["time_range"] = time_range.as_dict()
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def cached_zarr_path(
        self,
        variables: Sequence[str],
        *,
        region: RegionBounds | None = None,
        time_range: TimeRange | None = None,
    ) -> Path:
        key = self.cache_key(variables, region=region, time_range=time_range)
        return self.cache_dir / f"cf_{key}.zarr"

    def _subset_factual(
        self,
        variables: Sequence[str],
        *,
        region: RegionBounds | None,
        time_range: TimeRange | None,
        out_zarr: Path,
    ) -> Path:
        """Write a (possibly subset) factual Zarr for ATTRICI input."""
        ds = xr.open_zarr(self.factual_zarr, consolidated=True)
        lat_name = "latitude" if "latitude" in ds.dims or "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.dims or "longitude" in ds.coords else "lon"

        if region is not None:
            ds = ds.sel(
                {
                    lat_name: slice(region.lat_min, region.lat_max),
                    lon_name: slice(region.lon_min, region.lon_max),
                }
            )
        if time_range is not None and "time" in ds.coords:
            ds = ds.sel(time=slice(str(time_range.start_year), str(time_range.end_year)))

        keep = [v for v in normalize_variables(variables) if v in ds.data_vars]
        if not keep:
            raise KeyError(
                f"No requested variables in factual Zarr; wanted {variables}, "
                f"have {list(ds.data_vars)}"
            )
        out = ds[keep]
        if out_zarr.exists():
            shutil.rmtree(out_zarr)
        out.to_zarr(out_zarr, mode="w")
        return out_zarr

    def build_counterfactual_zarr(
        self,
        variables: Sequence[str] | None = None,
        *,
        region: RegionBounds | None = None,
        time_range: TimeRange | None = None,
        output_zarr: Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Run ATTRICI subprocess and return path to counterfactual Zarr (cached).

        Counterfactual variables are stored per ERA5 name; merge with factual and
        call :func:`data.attrici_fast_detrend.recompute_derived_counterfactuals` for
        ``vpd_mean_cf`` / ``et0_cf`` when needed.
        """
        vars_norm = normalize_variables(variables or ERA5_VARIABLES)
        out_path = output_zarr or self.cached_zarr_path(
            vars_norm, region=region, time_range=time_range
        )
        if out_path.exists() and not overwrite:
            log.info("Using cached ATTRICI counterfactual %s", out_path)
            return out_path

        if not self.factual_zarr.is_dir():
            raise FileNotFoundError(f"Factual ERA5 Zarr not found: {self.factual_zarr}")
        if not self.gmt_file.is_file():
            raise FileNotFoundError(
                f"GMT file for ATTRICI not found: {self.gmt_file}. "
                "Build via ATTRICI ``ssa`` or set ATTRICI_GMT_FILE."
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="attrici_cf_", dir=self.cache_dir))
        try:
            factual_subset = work / "factual_subset.zarr"
            self._subset_factual(vars_norm, region=region, time_range=time_range, out_zarr=factual_subset)

            runner = ATTRICIRunner(
                gmt_file=self.gmt_file,
                work_dir=work / "attrici_work",
                attrici_bin=self.attrici_bin,
                n_workers=self.n_workers,
                backend=self.backend,
            )
            runner.run(factual_subset, vars_norm, out_path, overwrite=True)

            # Merge factual + *_cf suffix for downstream 11-channel extraction
            self._finalize_merged_store(factual_subset, out_path, vars_norm)
            log.info("ATTRICI counterfactual ready at %s", out_path)
            return out_path
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _finalize_merged_store(
        self,
        factual_subset: Path,
        cf_zarr: Path,
        variables: Sequence[str],
    ) -> None:
        """Add ``{var}_cf`` arrays alongside factual variables in ``cf_zarr``."""
        xr = __import__("xarray", fromlist=["xarray"])
        fac = xr.open_zarr(factual_subset)
        cf_parts = load_counterfactual(cf_zarr)
        merged = fac.copy()
        for var in variables:
            if var in cf_parts.data_vars:
                merged[f"{var}_cf"] = cf_parts[var].rename(f"{var}_cf")
        try:
            merged = recompute_derived_counterfactuals(merged)
        except Exception as exc:
            log.warning("Derived counterfactual recompute skipped: %s", exc)
        if cf_zarr.exists():
            shutil.rmtree(cf_zarr)
        merged.attrs["counterfactual_method"] = "ATTRICI v2.0.1 subprocess"
        merged.attrs["reference"] = MENGEL_2021_REF
        merged.to_zarr(cf_zarr, mode="w")

    def open_counterfactual(self, path: Path | None = None, **kwargs: Any) -> xr.Dataset:
        """Open a built counterfactual store (default: cache lookup)."""
        zarr_path = path or self.build_counterfactual_zarr(**kwargs)
        return xr.open_zarr(zarr_path, consolidated=True)


__all__ = [
    "ATTRICICounterfactual",
    "ERA5_VARIABLES",
    "ISIMIP_ALIASES",
    "MENGEL_2021_REF",
    "RegionBounds",
    "TimeRange",
    "normalize_variables",
    "recompute_derived_counterfactuals",
]
