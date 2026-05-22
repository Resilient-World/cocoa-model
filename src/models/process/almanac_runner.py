"""
Python wrapper for the USDA-ARS ALMANAC Fortran executable (subprocess, no R).

ALMANAC: Agricultural Land Management Alternatives with Numerical Assessment Criteria.
Tropical tree-crop adaptations support cocoa/coffee (see Kiniry et al., *Agronomy* 2023).

Weather/soil/management files follow DSSAT-style ``*.WTH``, ``*.SOL``, and ``*.MGT``
conventions used in ALMANAC distributions. Daily plant output is read from ``*.PLN``;
annual summaries from ``*.OSR``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

USDA_ALMANAC_URL = (
    "https://www.ars.usda.gov/plains-area/temple-tx/"
    "grassland-soil-and-water-research-laboratory/docs/almanac/"
)
INSTALL_MESSAGE = (
    "ALMANAC executable not found.\n"
    f"Download and build ALMANAC from USDA-ARS: {USDA_ALMANAC_URL}\n"
    "Set ALMANAC_BINARY to the compiled binary path, install it on PATH as 'almanac', "
    "or pass almanac_binary= to ALMANACRunner.\n"
    "Optional: set ALMANAC_DATA_DIR to the parameter database shipped with the distribution."
)

REQUIRED_WEATHER_COLUMNS: tuple[str, ...] = (
    "date",
    "tmin_c",
    "tmax_c",
    "precip_mm",
    "srad_mj",
    "vapor_pressure_kpa",
)

T_MEAN_MIN_C = 10.0
T_MEAN_MAX_C = 40.0
ANNUAL_PRECIP_MIN_MM = 1250.0

DEFAULT_LAYER_DEPTHS_CM: tuple[float, ...] = (30.0, 60.0, 60.0)
DEFAULT_SAT: tuple[float, ...] = (0.43, 0.41, 0.39)
DEFAULT_FC: tuple[float, ...] = (0.32, 0.30, 0.28)
DEFAULT_WP: tuple[float, ...] = (0.15, 0.14, 0.13)
DEFAULT_CLAY: tuple[float, ...] = (25.0, 22.0, 20.0)
DEFAULT_SAND: tuple[float, ...] = (40.0, 42.0, 45.0)

_RUN_TIMEOUT_S = 600

# Column aliases in PLN / OSR tables (case-insensitive)
_PLN_LAI_ALIASES = ("LAI", "GLAI", "LAID")
_PLN_SW_ALIASES = ("SW", "SWAD", "SOILW")
_OSR_YIELD_ALIASES = ("HWAM", "YIELD", "GRAIN", "HWAD", "YLD")
_OSR_BIOMASS_ALIASES = ("CWAM", "CWAD", "BIOMASS", "PWAD", "TWAD")
_OSR_YEAR_ALIASES = ("YEAR", "YR", "FYEAR")


class ALMANACNotInstalled(RuntimeError):
    """Raised when the ALMANAC binary cannot be located or executed."""


@dataclass(frozen=True)
class ALMANACResult:
    """Structured output from an ALMANAC simulation run."""

    yearly_biomass_kg_ha: np.ndarray
    yearly_yield_kg_ha: np.ndarray
    daily_lai: np.ndarray
    daily_swc: np.ndarray  # shape (n_days, n_layers) when layer SW columns exist
    validity_flags: dict[str, bool] = field(default_factory=dict)


class ALMANACRunner:
    """
    Run ALMANAC via subprocess with DSSAT-style input files in a temporary directory.

    Parameters
    ----------
    almanac_binary:
        Path to the ALMANAC executable. Falls back to ``ALMANAC_BINARY`` env var, then
        ``shutil.which('almanac')``.
    data_dir:
        Optional directory of crop/soil parameter files from the ALMANAC distribution
        (copied into each run directory before execution).
    """

    def __init__(
        self,
        almanac_binary: str | None = None,
        data_dir: Path | None = None,
    ) -> None:
        binary = almanac_binary or os.environ.get("ALMANAC_BINARY")
        if binary:
            path = Path(binary).expanduser()
            if not path.is_file():
                raise ALMANACNotInstalled(f"ALMANAC binary not found: {path}\n{INSTALL_MESSAGE}")
            self._binary = path.resolve()
        else:
            found = shutil.which("almanac")
            if not found:
                raise ALMANACNotInstalled(INSTALL_MESSAGE)
            self._binary = Path(found).resolve()

        env_data = os.environ.get("ALMANAC_DATA_DIR")
        self._data_dir = Path(data_dir or env_data).expanduser() if (data_dir or env_data) else None
        if self._data_dir is not None and not self._data_dir.is_dir():
            raise FileNotFoundError(f"ALMANAC data_dir not found: {self._data_dir}")

    def simulate(
        self,
        weather: pd.DataFrame,
        soil: dict[str, Any],
        management: dict[str, Any],
        n_years: int = 8,
    ) -> ALMANACResult:
        """
        Write inputs, run ALMANAC, and parse ``.PLN`` / ``.OSR`` outputs.

        Weather columns must match ERA5 ingest naming (see :data:`REQUIRED_WEATHER_COLUMNS`).
        """
        weather_df = _prepare_weather(weather)
        validity = _validate_climate(weather_df)

        min_days = n_years * 365
        if len(weather_df) < min_days:
            raise ValueError(
                f"weather has {len(weather_df)} days; need at least ~{min_days} for {n_years} years."
            )

        station = str(management.get("station_id", "COCO")).upper()[:4]
        lat = float(management.get("latitude", 6.0))
        lon = float(management.get("longitude", -2.0))
        elev = float(management.get("elevation_m", 200.0))

        with tempfile.TemporaryDirectory(prefix="almanac_") as tmp:
            run_dir = Path(tmp)
            if self._data_dir is not None:
                _copy_data_dir(self._data_dir, run_dir)

            wth_path = run_dir / f"{station}.WTH"
            sol_path = run_dir / f"{station}.SOL"
            mgt_path = run_dir / f"{station}.MGT"

            _write_wth(wth_path, weather_df, station=station, lat=lat, lon=lon, elev=elev)
            _write_sol(sol_path, soil, station=station, lat=lat, lon=lon)
            _write_mgt(mgt_path, management, station=station, n_years=n_years)

            log.info("Running ALMANAC in %s", run_dir)
            proc = subprocess.run(
                [str(self._binary), "-i", str(run_dir)],
                check=False,
                capture_output=True,
                text=True,
                timeout=_RUN_TIMEOUT_S,
                cwd=str(run_dir),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ALMANAC exited with code {proc.returncode}.\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )

            pln_files = sorted(run_dir.glob("*.PLN")) + sorted(run_dir.glob("*.pln"))
            osr_files = sorted(run_dir.glob("*.OSR")) + sorted(run_dir.glob("*.osr"))
            if not pln_files:
                raise FileNotFoundError(
                    f"No .PLN output in {run_dir}. ALMANAC may use a different output naming convention."
                )

            daily = _parse_pln(pln_files[0])
            yearly = _parse_osr(osr_files[0]) if osr_files else _yearly_from_daily(daily, n_years)

            return ALMANACResult(
                yearly_biomass_kg_ha=yearly["biomass"],
                yearly_yield_kg_ha=yearly["yield"],
                daily_lai=daily["lai"],
                daily_swc=daily["swc"],
                validity_flags=validity,
            )

    @staticmethod
    def weather_from_era5_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Map ERA5-Land ingest columns to ALMANAC weather columns."""
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(
            df.index if isinstance(df.index, pd.DatetimeIndex) else df["time"]
        )
        out["tmin_c"] = df["tmin"].astype(float)
        out["tmax_c"] = df["tmax"].astype(float)
        out["precip_mm"] = df["precip"].astype(float)
        out["srad_mj"] = df["srad"].astype(float)
        if "vp_mean" in df.columns:
            out["vapor_pressure_kpa"] = df["vp_mean"].astype(float)
        elif "rh_mean" in df.columns and "tmean" in df.columns:
            tmean = df["tmean"].astype(float)
            rh = df["rh_mean"].astype(float).clip(0, 100)
            es = 0.61094 * np.exp(17.625 * tmean / (tmean + 243.04))
            out["vapor_pressure_kpa"] = es * rh / 100.0
        else:
            raise KeyError("ERA5 frame needs vp_mean or (rh_mean + tmean) for vapor pressure")
        return out.reset_index(drop=True)


def _copy_data_dir(src: Path, dest: Path) -> None:
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_WEATHER_COLUMNS if c not in weather.columns]
    if missing:
        raise ValueError(f"weather missing required columns: {missing}")
    df = weather.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in REQUIRED_WEATHER_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def _validate_climate(weather: pd.DataFrame) -> dict[str, bool]:
    tmean = (weather["tmin_c"] + weather["tmax_c"]) / 2.0
    mean_t = float(tmean.mean())
    flags: dict[str, bool] = {"mean_temperature_in_bounds": T_MEAN_MIN_C <= mean_t <= T_MEAN_MAX_C}
    if not flags["mean_temperature_in_bounds"]:
        raise ValueError(
            f"Mean temperature {mean_t:.2f} °C outside [{T_MEAN_MIN_C}, {T_MEAN_MAX_C}] °C."
        )
    annual_precip = weather.groupby(weather["date"].dt.year)["precip_mm"].sum()
    low_years = annual_precip[annual_precip < ANNUAL_PRECIP_MIN_MM]
    flags["annual_precipitation_adequate"] = low_years.empty
    if not flags["annual_precipitation_adequate"]:
        warnings.warn(
            f"Annual precipitation below {ANNUAL_PRECIP_MIN_MM} mm in year(s) {list(low_years.index)}.",
            stacklevel=2,
        )
    return flags


def _yyddd(dt: pd.Timestamp) -> str:
    return f"{dt.year % 100:02d}{dt.dayofyear:03d}"


def _write_wth(
    path: Path,
    weather: pd.DataFrame,
    *,
    station: str,
    lat: float,
    lon: float,
    elev: float,
) -> None:
    """Write DSSAT-style daily weather (``.WTH``)."""
    tmean = (weather["tmin_c"] + weather["tmax_c"]) / 2.0
    tav = float(tmean.mean())
    amp = float(weather["tmax_c"].max() - weather["tmin_c"].min()) / 2.0

    lines = [
        f"*WEATHER DATA : {station} (generated by resilient-cocoa-model)",
        "",
        "@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT",
        f"  {station:<4} {lat:7.3f} {lon:8.3f} {elev:5.0f} {tav:5.1f} {amp:5.1f}   2.0   2.0",
        "",
        "@DATE  SRAD  TMAX  TMIN  RAIN",
    ]
    for _, row in weather.iterrows():
        dt = pd.Timestamp(row["date"])
        lines.append(
            f"{_yyddd(dt):5s} {row['srad_mj']:5.1f} {row['tmax_c']:5.1f} "
            f"{row['tmin_c']:5.1f} {row['precip_mm']:5.1f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _fill_layer(
    values: list[float] | None,
    default: tuple[float, ...],
    n_layers: int,
    name: str,
) -> list[float]:
    if values is None:
        return list(default[:n_layers])
    if len(values) != n_layers:
        raise ValueError(f"soil[{name}] must have {n_layers} values, got {len(values)}")
    return [float(v) for v in values]


def _write_sol(
    path: Path,
    soil: dict[str, Any],
    *,
    station: str,
    lat: float,
    lon: float,
) -> None:
    """Write DSSAT-style soil profile (``.SOL``)."""
    depths = soil.get("layer_depths_cm", list(DEFAULT_LAYER_DEPTHS_CM))
    depths = [float(d) for d in depths]
    if abs(sum(depths) - 150.0) > 1.0:
        raise ValueError(f"layer_depths_cm should sum to ~150 cm, got {sum(depths)}")
    n = len(depths)
    cum = np.cumsum(depths)
    sat = _fill_layer(soil.get("sat"), DEFAULT_SAT, n, "sat")
    fc = _fill_layer(soil.get("fc"), DEFAULT_FC, n, "fc")
    wp = _fill_layer(soil.get("wp"), DEFAULT_WP, n, "wp")
    clay = _fill_layer(soil.get("clay_pct"), DEFAULT_CLAY, n, "clay_pct")
    sand = _fill_layer(soil.get("sand_pct"), DEFAULT_SAND, n, "sand_pct")
    soil_id = str(soil.get("soil_id", f"{station}001"))

    lines = [
        f"*SOIL DATA : {station} profile",
        "",
        "@SITE   LAT     LONG SCS FAMILY",
        f" {soil_id:<6} {lat:7.3f} {lon:8.3f}  -99  Generic",
        "",
        "@ SLB  SLCL  SLSI  SLCF  SLBD  SLHW  SLSAT  SLDR  SLRO  SLNI",
    ]
    for i in range(n):
        silt = max(0.0, 100.0 - clay[i] - sand[i])
        lines.append(
            f" {cum[i]:4.0f} {clay[i]:5.1f} {silt:5.1f} {sand[i]:5.1f} "
            f" 1.35  6.0  {sat[i]:5.3f}  0.20  75.0   0.0"
        )
    lines.extend(
        [
            "",
            "@ SLB  SDUL  SSAT  SRGF  SSKS  SBDM  SLCL  SLSI",
        ]
    )
    for i in range(n):
        lines.append(
            f" {cum[i]:4.0f} {fc[i]:5.3f} {sat[i]:5.3f}  1.00  2.00  1.35 "
            f"{clay[i]:5.1f} {max(0.0, 100.0 - clay[i] - sand[i]):5.1f}"
        )
    lines.extend(["", f"* wilting point fractions (internal): {wp}"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_mgt(
    path: Path,
    management: dict[str, Any],
    *,
    station: str,
    n_years: int,
) -> None:
    """Write DSSAT-style management (``.MGT``)."""
    crop = str(management.get("crop_code", "CC"))[:2]
    pop = int(management.get("planting_density", 1100))
    plant_doy = int(management.get("planting_doy", 152))
    name = str(management.get("field_name", "COCOA"))[:16]

    lines = [
        f"*MANAGEMENT : {station} {n_years}-year run",
        "",
        "@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM",
        f" 1 1 0 0 {name:<16} {crop} CO IB  1  1  0  0  0  0  0  0  0  0  1",
        "",
        "@N PLANTING    PDP     PLPOP  PLME PLDS",
        f" 1 {plant_doy:7d} IB     {pop:6d}   S     R",
        "",
        f"* Simulation years requested: {n_years}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _parse_dssat_table(path: Path) -> pd.DataFrame:
    """Parse DSSAT/ALMANAC columnar tables (``@`` header rows)."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx: int | None = None
    columns: list[str] = []
    for i, line in enumerate(text):
        if line.startswith("@"):
            parts = line.lstrip("@").split()
            if len(parts) >= 2 and parts[0].isalpha():
                header_idx = i
                columns = parts
    if header_idx is None or not columns:
        raise ValueError(f"No @ header table found in {path}")

    rows: list[list[float]] = []
    for line in text[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("!"):
            continue
        if stripped.startswith("@"):
            break
        parts = stripped.split()
        if not parts[0][:1].isdigit():
            continue
        try:
            rows.append([float(p) for p in parts[: len(columns)]])
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")
    return pd.DataFrame(rows, columns=columns)


def _find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    upper = {c.upper(): c for c in df.columns}
    for alias in aliases:
        if alias in upper:
            return upper[alias]
    return None


def _parse_pln(path: Path) -> dict[str, np.ndarray]:
    df = _parse_dssat_table(path)
    lai_col = _find_column(df, _PLN_LAI_ALIASES)
    if lai_col is None:
        raise ValueError(f"LAI column not found in {path}; columns={list(df.columns)}")

    lai = df[lai_col].to_numpy(dtype=np.float64)

    sw_cols = [
        c for c in df.columns if re.match(r"SW\d*", c.upper()) or c.upper() in _PLN_SW_ALIASES
    ]
    if not sw_cols:
        sw_cols = [c for c in df.columns if c.upper().startswith("SW") and c.upper() != "SW"]
    if sw_cols:
        swc = df[sw_cols].to_numpy(dtype=np.float64)
    else:
        swc = np.full((len(lai), 1), np.nan, dtype=np.float64)

    return {"lai": lai, "swc": swc, "frame": df}


def _parse_osr(path: Path) -> dict[str, np.ndarray]:
    df = _parse_dssat_table(path)
    y_col = _find_column(df, _OSR_YEAR_ALIASES)
    yield_col = _find_column(df, _OSR_YIELD_ALIASES)
    biomass_col = _find_column(df, _OSR_BIOMASS_ALIASES)
    if yield_col is None and biomass_col is None:
        raise ValueError(f"Yield/biomass columns not found in {path}; columns={list(df.columns)}")

    if y_col is not None:
        df = df.sort_values(y_col)

    yield_arr = df[yield_col].to_numpy(dtype=np.float64) if yield_col else np.full(len(df), np.nan)
    biomass_arr = df[biomass_col].to_numpy(dtype=np.float64) if biomass_col else yield_arr.copy()
    return {"yield": yield_arr, "biomass": biomass_arr}


def _yearly_from_daily(daily: dict[str, Any], n_years: int) -> dict[str, np.ndarray]:
    """Fallback when ``.OSR`` is absent: coarse annual placeholders from daily LAI."""
    n_days = len(daily["lai"])
    years = max(1, n_days // 365)
    years = min(years, n_years)
    lai = daily["lai"]
    approx_yield = np.array(
        [float(np.nanmax(lai[i * 365 : (i + 1) * 365]) * 100.0) for i in range(years)],
        dtype=np.float64,
    )
    return {"yield": approx_yield, "biomass": approx_yield * 1.5}
