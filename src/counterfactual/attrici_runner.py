"""
Counterfactual climate via ATTRICI (subprocess only — no ``import attrici``).

Distributions per Mengel et al. 2021 (GMD 14, 5269) Table 1 are encoded in ATTRICI
by variable name; :data:`ATTRICI_DISTRIBUTIONS` documents that mapping for review.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Mengel et al. 2021, GMD 14, 5269 — Table 1 (§3.2)
ATTRICI_DISTRIBUTIONS: dict[str, tuple[str, str]] = {
    "tas": ("normal", "identity"),
    "tasrange": ("gamma", "log"),
    "tasskew": ("normal", "identity"),
    "pr": ("bernoulli_gamma", "log"),
    "rsds": ("normal", "identity"),
    "sfcwind": ("weibull", "log"),
    "hurs": ("beta", "logit"),
}

# ATTRICI v2.0.1 internal names (see attrici.variables.create_variable)
_ATTRICI_VAR_ALIASES: dict[str, str] = {
    "sfcwind": "sfcWind",
}

# Detrended in ISIMIP3a / ATTRICI workflow (huss derived post-hoc; ps kept factual)
_DETREND_TARGETS: tuple[str, ...] = (
    "tas",
    "tasrange",
    "tasskew",
    "pr",
    "hurs",
    "rsds",
    "sfcWind",
)

_KELVIN_THRESHOLD = 150.0


@dataclass
class ATTRICIConfig:
    variables: tuple[str, ...] = (
        "tas",
        "tasmin",
        "tasmax",
        "pr",
        "hurs",
        "rsds",
        "sfcwind",
    )
    aoi_bbox: tuple[float, float, float, float] = (-10.0, 4.0, 14.0, 11.0)  # W, S, E, N
    start_year: int = 1979
    end_year: int = 2019
    backend: str = "scipy"  # "scipy" | "pymc5"
    ssa_window_years: int = 10
    n_fourier_modes: int = 4
    attrici_venv: Path = Path(".venv-attrici")
    work_dir: Path = Path("data/attrici_work")
    gmt_file: Path = Path("data/raw/gmt/ssa_gmt.nc")
    counterfactual_dir: Path = Path("data/counterfactual")
    factual_ps_var: str = "ps"


@dataclass
class CellJob:
    variable: str
    lat: float
    lon: float


@dataclass
class CellJobResult:
    variable: str
    lat: float
    lon: float
    returncode: int
    stderr: str = ""


def _attrici_bin(config: ATTRICIConfig) -> Path:
    return config.attrici_venv / "bin" / "attrici"


def _shim_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "attrici_cli_shim.py"


def _to_attrici_var(name: str) -> str:
    return _ATTRICI_VAR_ALIASES.get(name, name)


def _normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    if "lon" in ds.coords and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.sortby("lon")
    return ds


def _subset_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    west, south, east, north = bbox
    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"
    ds = _normalize_lon(ds)
    return ds.sel(
        **{
            lat_name: slice(south, north),
            lon_name: slice(west, east),
        }
    )


def _subset_years(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    return ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


def _tas_celsius(da: xr.DataArray) -> xr.DataArray:
    """Return temperature in °C (W5E5/ISIMIP often store tas in K)."""
    if float(da.mean(skipna=True)) > _KELVIN_THRESHOLD:
        return da - 273.15
    return da


def _derive_tasrange_tasskew(
    tas: xr.DataArray,
    tasmin: xr.DataArray,
    tasmax: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    tas_c = _tas_celsius(tas)
    tmin_c = _tas_celsius(tasmin)
    tmax_c = _tas_celsius(tasmax)
    tasrange = (tmax_c - tmin_c).rename("tasrange")
    with np.errstate(divide="ignore", invalid="ignore"):
        tasskew = ((tas_c - tmin_c) / tasrange).rename("tasskew")
    tasskew = tasskew.where(tasrange > 0)
    return tasrange, tasskew


def _postprocess_tasmin_tasmax(
    tas: xr.DataArray,
    tasrange: xr.DataArray,
    tasskew: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Inverse of ATTRICI preprocess-tas (CDO formulas)."""
    tas_c = _tas_celsius(tas)
    tasmin = (tas_c - tasskew * tasrange).rename("tasmin")
    tasmax = (tasmin + tasrange).rename("tasmax")
    return tasmin, tasmax


def buck_saturation_vapor_pressure_hpa(temperature_c: float | xr.DataArray) -> float | xr.DataArray:
    """Saturation vapor pressure (hPa) from Buck (1981); ``temperature_c`` in °C."""
    t = temperature_c
    return 6.1121 * np.exp((18.678 - t / 234.5) * t / (257.14 + t))


def _buck_huss(
    tas_c: xr.DataArray,
    hurs_pct: xr.DataArray,
    ps_pa: xr.DataArray,
) -> xr.DataArray:
    """
    Specific humidity from Buck (1981) via Mengel §3.2.7 / Weedon (2010).

    ``e_s`` in hPa; ``ps`` in Pa; ``hurs`` in percent.
    """
    e_s = buck_saturation_vapor_pressure_hpa(tas_c)
    e_hpa = e_s * (hurs_pct / 100.0)
    ps_hpa = ps_pa / 100.0
    return (0.622 * e_hpa / (ps_hpa - 0.378 * e_hpa)).rename("huss")


def _write_isimip_variable(ds: xr.Dataset, var: str, path: Path) -> None:
    if var not in ds:
        raise KeyError(f"Variable {var!r} not in dataset; have {list(ds.data_vars)}")
    out = ds[[var]].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)


def _config_from_dict(data: dict[str, Any]) -> ATTRICIConfig:
    path_fields = ("attrici_venv", "work_dir", "gmt_file", "counterfactual_dir")
    kwargs = dict(data)
    for key in path_fields:
        if key in kwargs:
            kwargs[key] = Path(kwargs[key])
    return ATTRICIConfig(**kwargs)


def _run_cell_subprocess(
    job: CellJob, runner_paths: dict[str, str], config_dict: dict[str, Any]
) -> CellJobResult:
    config = _config_from_dict(config_dict)
    attrici_var = job.variable
    input_file = Path(runner_paths["input_dir"]) / f"{attrici_var}.nc"

    cmd = [
        sys.executable,
        str(_shim_script()),
        "--attrici-bin",
        str(_attrici_bin(config)),
        "--gmt-file",
        str(config.gmt_file),
        "--input-file",
        str(input_file),
        "--output-dir",
        str(Path(runner_paths["detrend_dir"])),
        "--variable",
        attrici_var,
        "--lat",
        str(job.lat),
        "--lon",
        str(job.lon),
        "--modes",
        str(config.n_fourier_modes),
        "--solver",
        config.backend,
        "--start-date",
        f"{config.start_year}-01-01",
        "--stop-date",
        f"{config.end_year}-12-31",
        "--overwrite",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return CellJobResult(
        variable=job.variable,
        lat=job.lat,
        lon=job.lon,
        returncode=result.returncode,
        stderr=result.stderr[-2000:] if result.stderr else "",
    )


class CounterfactualClimateRunner:
    """Run ATTRICI detrending in an isolated venv via subprocess per grid cell."""

    def __init__(self, config: ATTRICIConfig) -> None:
        self.config = config
        self._factual_ds: xr.Dataset | None = None
        self._input_dir: Path = config.work_dir / "input"
        self._detrend_dir: Path = config.work_dir / "detrend"
        self._merged_dir: Path = config.work_dir / "merged"
        self._failed_cells: list[CellJobResult] = []

    @property
    def failed_cells(self) -> list[CellJobResult]:
        return list(self._failed_cells)

    def prepare_inputs(self, factual_ds: xr.Dataset) -> Path:
        """Write factual NetCDF in ISIMIP3a layout (one file per variable)."""
        cfg = self.config
        ds = _subset_years(_subset_bbox(factual_ds, cfg.aoi_bbox), cfg.start_year, cfg.end_year)
        self._factual_ds = ds
        self._input_dir.mkdir(parents=True, exist_ok=True)

        written: set[str] = set()
        for var in cfg.variables:
            attrici_name = _to_attrici_var(var)
            _write_isimip_variable(ds, var, self._input_dir / f"{attrici_name}.nc")
            written.add(attrici_name)

        if all(v in ds for v in ("tas", "tasmin", "tasmax")):
            tasrange, tasskew = _derive_tasrange_tasskew(ds["tas"], ds["tasmin"], ds["tasmax"])
            xr.Dataset({"tasrange": tasrange, "tasskew": tasskew}).to_netcdf(
                self._input_dir / "_tas_derived.nc"
            )
            tasrange.to_netcdf(self._input_dir / "tasrange.nc")
            tasskew.to_netcdf(self._input_dir / "tasskew.nc")
            written.update({"tasrange", "tasskew"})

        if cfg.factual_ps_var in ds and cfg.factual_ps_var not in written:
            _write_isimip_variable(ds, cfg.factual_ps_var, self._input_dir / f"{cfg.factual_ps_var}.nc")

        logger.info("Prepared ATTRICI inputs in %s (%d variables)", self._input_dir, len(written))
        return self._input_dir

    def _detrend_variables(self) -> list[str]:
        """ATTRICI variable names present in the input directory."""
        found: list[str] = []
        for v in _DETREND_TARGETS:
            if (self._input_dir / f"{v}.nc").is_file():
                found.append(v)
        return found

    def _grid_jobs(self) -> list[CellJob]:
        if self._factual_ds is None:
            raise RuntimeError("Call prepare_inputs() before run()")
        lat_name = "lat" if "lat" in self._factual_ds.dims else "latitude"
        lon_name = "lon" if "lon" in self._factual_ds.dims else "longitude"
        lats = [float(v) for v in self._factual_ds[lat_name].values]
        lons = [float(v) for v in self._factual_ds[lon_name].values]

        jobs: list[CellJob] = []
        for var in self._detrend_variables():
            for lat in lats:
                for lon in lons:
                    jobs.append(CellJob(variable=var, lat=lat, lon=lon))
        return jobs

    def run(self) -> Path:
        """Invoke ATTRICI via subprocess per (variable, gridcell). Returns counterfactual NetCDF dir."""
        cfg = self.config
        if not _attrici_bin(cfg).exists():
            raise FileNotFoundError(
                f"ATTRICI venv not found at {cfg.attrici_venv}. Run: make attrici-env"
            )
        if not cfg.gmt_file.is_file():
            raise FileNotFoundError(
                f"SSA-smoothed GMT file required at {cfg.gmt_file}. "
                "See docs/data/gswp3_w5e5.md"
            )

        self._detrend_dir.mkdir(parents=True, exist_ok=True)
        jobs = self._grid_jobs()
        if not jobs:
            raise RuntimeError("No detrend jobs; check prepare_inputs() and variable files")

        runner_paths = {
            "input_dir": str(self._input_dir),
            "detrend_dir": str(self._detrend_dir),
        }
        config_dict: dict[str, Any] = {
            "variables": cfg.variables,
            "aoi_bbox": cfg.aoi_bbox,
            "start_year": cfg.start_year,
            "end_year": cfg.end_year,
            "backend": cfg.backend,
            "ssa_window_years": cfg.ssa_window_years,
            "n_fourier_modes": cfg.n_fourier_modes,
            "attrici_venv": str(cfg.attrici_venv),
            "work_dir": str(cfg.work_dir),
            "gmt_file": str(cfg.gmt_file),
            "counterfactual_dir": str(cfg.counterfactual_dir),
            "factual_ps_var": cfg.factual_ps_var,
        }

        n_proc = os.cpu_count() or 1
        logger.info("Running %d ATTRICI cell jobs with %d workers", len(jobs), n_proc)
        with Pool(processes=n_proc) as pool:
            results = pool.starmap(
                _run_cell_subprocess,
                [(job, runner_paths, config_dict) for job in jobs],
            )

        self._failed_cells = [r for r in results if r.returncode != 0]
        if self._failed_cells:
            fail_log = cfg.work_dir / "failed_cells.jsonl"
            with fail_log.open("w", encoding="utf-8") as fh:
                for r in self._failed_cells:
                    fh.write(
                        json.dumps(
                            {
                                "variable": r.variable,
                                "lat": r.lat,
                                "lon": r.lon,
                                "returncode": r.returncode,
                                "stderr": r.stderr,
                            }
                        )
                        + "\n"
                    )
            logger.warning("%d cells failed; see %s", len(self._failed_cells), fail_log)

        self._merge_detrend_outputs()
        self._postprocess_and_publish()
        return cfg.counterfactual_dir

    def _merge_variable(self, attrici_var: str, out_name: str) -> Path | None:
        ts_dir = self._detrend_dir / "timeseries" / attrici_var
        if not ts_dir.is_dir():
            logger.warning("No detrend output for %s at %s", attrici_var, ts_dir)
            return None
        merged_path = self._merged_dir / f"{out_name}_cfact.nc"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(_attrici_bin(self.config)),
            "merge-output",
            str(ts_dir),
            str(merged_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return merged_path

    def _read_merged_cfact(self, path: Path, var_name: str) -> xr.DataArray:
        ds = xr.open_dataset(path)
        if "cfact" in ds:
            da = ds["cfact"].rename(var_name)
        elif var_name in ds:
            da = ds[var_name]
        else:
            da = ds[list(ds.data_vars)[0]].rename(var_name)
        ds.close()
        return da

    def _merge_detrend_outputs(self) -> None:
        self._merged_dir.mkdir(parents=True, exist_ok=True)
        for var in self._detrend_variables():
            store = "sfcwind" if var == "sfcWind" else var
            self._merge_variable(var, store)

    def _postprocess_and_publish(self) -> None:
        cfg = self.config
        out_dir = cfg.counterfactual_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        merged: dict[str, xr.DataArray] = {}
        for var in _DETREND_TARGETS:
            store = "sfcwind" if var == "sfcWind" else var
            path = self._merged_dir / f"{store}_cfact.nc"
            if path.is_file():
                merged[store] = self._read_merged_cfact(path, store)

        if all(k in merged for k in ("tas", "tasrange", "tasskew")):
            tasmin, tasmax = _postprocess_tasmin_tasmax(
                merged["tas"], merged["tasrange"], merged["tasskew"]
            )
            merged["tasmin"] = tasmin
            merged["tasmax"] = tasmax

        if "rsds" in merged:
            merged["rsds"] = merged["rsds"].clip(min=0.0)

        if self._factual_ds is not None and cfg.factual_ps_var in self._factual_ds:
            ps = self._factual_ds[cfg.factual_ps_var]
            if "tas" in merged and "hurs" in merged:
                merged["huss"] = _buck_huss(
                    _tas_celsius(merged["tas"]),
                    merged["hurs"],
                    ps,
                )

        if merged:
            xr.Dataset(merged).to_netcdf(out_dir / "counterfactual_merged.nc")
            for name, da in merged.items():
                da.to_dataset(name=name).to_netcdf(out_dir / f"{name}.nc")

    def load_counterfactual(self) -> xr.Dataset:
        """Read counterfactual NetCDFs back into a single xarray.Dataset."""
        cfg = self.config
        merged = cfg.counterfactual_dir / "counterfactual_merged.nc"
        if merged.is_file():
            return xr.open_dataset(merged)

        parts: dict[str, xr.DataArray] = {}
        for path in sorted(cfg.counterfactual_dir.glob("*.nc")):
            if path.name == "counterfactual_merged.nc":
                continue
            ds = xr.open_dataset(path)
            var = path.stem
            parts[var] = ds[var] if var in ds else ds[list(ds.data_vars)[0]]
            ds.close()
        if not parts:
            raise FileNotFoundError(f"No counterfactual NetCDFs in {cfg.counterfactual_dir}")
        return xr.Dataset(parts)


def ensure_gmt_ssa(
    raw_gmt: Path,
    output: Path,
    *,
    attrici_venv: Path,
    variable: str = "tas",
    window_days: int | None = None,
    subset: int | None = None,
) -> Path:
    """SSA-smooth a GMT series NetCDF (ATTRICI ``ssa`` subcommand)."""
    window_days = window_days or 365
    subset = subset or 10
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(attrici_venv / "bin" / "attrici"),
        "ssa",
        str(raw_gmt),
        str(output),
        "--variable",
        variable,
        "--window-size",
        str(window_days),
        "--subset",
        str(subset),
    ]
    subprocess.run(cmd, check=True)
    return output
