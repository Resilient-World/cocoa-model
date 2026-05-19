"""
Python wrapper for RCASE2 (Wageningen R interface to the FORTRAN CASE2 engine).

CASE2: Zuidema et al. (2005), *Agricultural Systems*; applied in Asante et al. (2022).
RCASE2 is distributed via WUR eDepot — see :exc:`CASE2NotInstalled` for install pointers.

``rpy2`` is imported only inside :meth:`CASE2Runner.__init__` so the module loads without R.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EDEPOT_URL = "https://edepot.wur.nl/16613"
INSTALL_MESSAGE = (
    "RCASE2 / CASE2 is not available in this environment.\n"
    f"Download the RCASE2 R package and CASE2 binaries from WUR eDepot: {EDEPOT_URL}\n"
    "Install R (>= 4.0), set R_HOME if needed, install rpy2 in Python, then install RCASE2 "
    "into your R library (e.g. R CMD INSTALL rcase2_*.tar.gz).\n"
    "Optional: set RCASE2_LIB_PATH to the directory containing the installed RCASE2 package."
)

REQUIRED_WEATHER_COLUMNS: tuple[str, ...] = (
    "date",
    "tmin_c",
    "tmax_c",
    "precip_mm",
    "srad_mj",
    "vapor_pressure_kpa",
)

# Asante et al. 2022, Agric. Syst. 201 — CASE2 validity (Sec. 2.1)
T_MEAN_MIN_C = 10.0
T_MEAN_MAX_C = 40.0
ANNUAL_PRECIP_MIN_MM = 1250.0

TREE_AGE_MIN_Y = 3
TREE_AGE_MAX_Y = 40
PLANTING_DENSITY_MIN = 700
PLANTING_DENSITY_MAX = 2500
SLAI_MIN = 0.0
SLAI_MAX = 3.0
DEFAULT_K_EXTINCTION = 0.6

# Driessen (1986) medium tropical profile — three layers summing to 150 cm
DEFAULT_LAYER_DEPTHS_CM: tuple[float, ...] = (50.0, 50.0, 50.0)
DEFAULT_SAT: tuple[float, ...] = (0.43, 0.43, 0.41)
DEFAULT_FC: tuple[float, ...] = (0.32, 0.32, 0.31)
DEFAULT_WP: tuple[float, ...] = (0.15, 0.15, 0.14)
DEFAULT_AD: tuple[float, ...] = (0.05, 0.05, 0.05)

# RCASE2 R entry points tried in order (package may expose one of these)
_R_SIMULATE_FUNCTIONS: tuple[str, ...] = (
    "run_case2",
    "case2_simulate",
    "simulate_case2",
)


class CASE2NotInstalled(RuntimeError):
    """Raised when R, rpy2, or the RCASE2 package/binary cannot be loaded."""


@dataclass(frozen=True)
class CASE2Result:
    """Structured output from a CASE2 / RCASE2 simulation run."""

    yearly_yield_kg_ha: np.ndarray
    daily_lai: np.ndarray
    daily_water_stress: np.ndarray
    daily_assimilation: np.ndarray
    validity_flags: dict[str, bool] = field(default_factory=dict)


class CASE2Runner:
    """
    Run RCASE2 via ``rpy2`` with pandas weather input and dict-based soil/management.

    Parameters
    ----------
    rcase2_lib_path:
        Directory containing the installed RCASE2 R package, or path to ``rcase2`` sources.
        Falls back to ``RCASE2_LIB_PATH`` then a standard R library search.
    r_home:
        R installation root (sets ``R_HOME`` before initializing rpy2).
    """

    def __init__(
        self,
        rcase2_lib_path: str | None = None,
        r_home: str | None = None,
    ) -> None:
        if r_home:
            os.environ["R_HOME"] = r_home

        try:
            import rpy2.robjects as ro  # noqa: PLC0415
            from rpy2.robjects import pandas2ri  # noqa: PLC0415
            from rpy2.robjects.conversion import localconverter  # noqa: PLC0415
            from rpy2.robjects.packages import importr  # noqa: PLC0415
        except ImportError as exc:
            raise CASE2NotInstalled(
                f"Python package rpy2 is not installed ({exc}).\n{INSTALL_MESSAGE}"
            ) from exc

        self._ro = ro
        self._pandas2ri = pandas2ri
        self._localconverter = localconverter
        self._importr = importr
        self._rcase2: Any = None
        self._simulate_fn: Any = None

        lib_path = rcase2_lib_path or os.environ.get("RCASE2_LIB_PATH")
        self._rcase2 = self._load_rcase2(lib_path)
        self._simulate_fn = self._resolve_simulate_function()

    def _load_rcase2(self, lib_path: str | None) -> Any:
        ro = self._ro
        if lib_path:
            path = Path(lib_path).expanduser().resolve()
            if not path.exists():
                raise CASE2NotInstalled(f"RCASE2 path does not exist: {path}\n{INSTALL_MESSAGE}")
            init_r = path / "R" / "rcase2-init.R"
            if init_r.is_file():
                ro.r(f'source("{init_r.as_posix()}")')
            elif (path / "DESCRIPTION").is_file():
                ro.r(f'library(rcase2, lib.loc="{path.as_posix()}")')
            else:
                ro.r(f'source("{path.as_posix()}")')

        try:
            return self._importr("rcase2")
        except Exception as exc:
            raise CASE2NotInstalled(
                f"Could not load R package 'rcase2' ({exc}).\n{INSTALL_MESSAGE}"
            ) from exc

    def _resolve_simulate_function(self) -> Any:
        for name in _R_SIMULATE_FUNCTIONS:
            try:
                return getattr(self._rcase2, name)
            except AttributeError:
                continue
        raise CASE2NotInstalled(
            "RCASE2 package loaded but no simulate function found "
            f"(tried: {', '.join(_R_SIMULATE_FUNCTIONS)}).\n{INSTALL_MESSAGE}"
        )

    def simulate(
        self,
        weather: pd.DataFrame,
        soil: dict[str, Any],
        management: dict[str, Any],
        n_years: int = 8,
    ) -> CASE2Result:
        """
        Run CASE2 for ``n_years`` using daily weather and site parameters.

        Weather must include: ``date``, ``tmin_c``, ``tmax_c``, ``precip_mm``,
        ``srad_mj``, ``vapor_pressure_kpa`` (ERA5-Land ingest naming).
        """
        weather_df = _prepare_weather(weather)
        validity = _validate_climate(weather_df)
        soil_r = self._build_soil_list(soil)
        mgmt_r = self._build_management_list(management)

        min_days = n_years * 365
        if len(weather_df) < min_days:
            raise ValueError(
                f"weather has {len(weather_df)} days; CASE2 requires at least "
                f"{n_years} years (~{min_days} days, Asante et al. 2022)."
            )

        with self._localconverter(
            self._ro.default_converter + self._pandas2ri.converter
        ):
            r_weather = self._ro.conversion.py2rpy(weather_df)
            r_result = self._simulate_fn(
                weather=r_weather,
                soil=soil_r,
                management=mgmt_r,
                n_years=int(n_years),
            )

        return _parse_r_result(r_result, validity, self._ro)

    def _build_soil_list(self, soil: dict[str, Any]) -> Any:
        depths = soil.get("layer_depths_cm", list(DEFAULT_LAYER_DEPTHS_CM))
        if isinstance(depths, (int, float)):
            depths = [float(depths)]
        depths = [float(d) for d in depths]
        if abs(sum(depths) - 150.0) > 1e-3:
            raise ValueError(f"layer_depths_cm must sum to 150 cm, got {sum(depths)}")
        n = len(depths)

        sat = _fill_layer(soil.get("sat"), DEFAULT_SAT, n, "sat")
        fc = _fill_layer(soil.get("fc"), DEFAULT_FC, n, "fc")
        wp = _fill_layer(soil.get("wp"), DEFAULT_WP, n, "wp")
        ad = _fill_layer(soil.get("ad"), DEFAULT_AD, n, "ad")

        return self._ro.ListVector(
            {
                "layer_depths_cm": self._ro.FloatVector(depths),
                "sat": self._ro.FloatVector(sat),
                "fc": self._ro.FloatVector(fc),
                "wp": self._ro.FloatVector(wp),
                "ad": self._ro.FloatVector(ad),
            }
        )

    def _build_management_list(self, management: dict[str, Any]) -> Any:
        age = float(management["tree_age_years"])
        density = float(management["planting_density"])
        slai = float(management["slai"])
        k_ext = float(management.get("k_extinction", DEFAULT_K_EXTINCTION))

        if not TREE_AGE_MIN_Y <= age <= TREE_AGE_MAX_Y:
            raise ValueError(f"tree_age_years must be in [{TREE_AGE_MIN_Y}, {TREE_AGE_MAX_Y}]")
        if not PLANTING_DENSITY_MIN <= density <= PLANTING_DENSITY_MAX:
            raise ValueError(
                f"planting_density must be in [{PLANTING_DENSITY_MIN}, {PLANTING_DENSITY_MAX}]"
            )
        if not SLAI_MIN <= slai <= SLAI_MAX:
            raise ValueError(f"slai must be in [{SLAI_MIN}, {SLAI_MAX}]")

        return self._ro.ListVector(
            {
                "tree_age_years": age,
                "planting_density": density,
                "slai": slai,
                "k_extinction": k_ext,
            }
        )

    @staticmethod
    def weather_from_era5_frame(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map ERA5 ingest column names to CASE2 weather columns.

        Expects ``tmin``, ``tmax``, ``precip``, ``srad``, and either ``vp_mean`` or
        ``rh_mean`` (+ ``tmean`` for vapor pressure derivation).
        """
        out = pd.DataFrame(index=df.index)
        out["date"] = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df["time"])
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
    flags: dict[str, bool] = {
        "mean_temperature_in_bounds": T_MEAN_MIN_C <= mean_t <= T_MEAN_MAX_C,
    }

    if not flags["mean_temperature_in_bounds"]:
        raise ValueError(
            f"Mean temperature {mean_t:.2f} °C outside CASE2 bounds "
            f"[{T_MEAN_MIN_C}, {T_MEAN_MAX_C}] °C (Asante et al. 2022)."
        )

    annual_precip = weather.groupby(weather["date"].dt.year)["precip_mm"].sum()
    low_years = annual_precip[annual_precip < ANNUAL_PRECIP_MIN_MM]
    flags["annual_precipitation_adequate"] = low_years.empty
    if not flags["annual_precipitation_adequate"]:
        warnings.warn(
            f"Annual precipitation below {ANNUAL_PRECIP_MIN_MM} mm in year(s) "
            f"{list(low_years.index)} — CASE2 may be unreliable (warn only).",
            stacklevel=2,
        )
    return flags


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


def _parse_r_result(r_result: Any, validity_flags: dict[str, bool], ro: Any) -> CASE2Result:
    """Convert RCASE2 R list output into :class:`CASE2Result`."""
    names = list(r_result.names) if r_result.names is not None else []

    def _arr(key: str, *aliases: str) -> np.ndarray:
        for k in (key, *aliases):
            if k in names:
                return np.asarray(ro.conversion.rpy2py(r_result.rx2[k]), dtype=np.float64)
        return np.array([], dtype=np.float64)

    yearly = _arr("yearly_yield_kg_ha", "yield_kg_ha", "bean_yield")
    lai = _arr("daily_lai", "lai")
    wstress = _arr("daily_water_stress", "water_stress", "wstress")
    assim = _arr("daily_assimilation", "assimilation", "assim")

    if yearly.size == 0:
        raise RuntimeError(
            "RCASE2 returned no yearly yield; check RCASE2 version and simulate function output."
        )

    return CASE2Result(
        yearly_yield_kg_ha=yearly,
        daily_lai=lai,
        daily_water_stress=wstress,
        daily_assimilation=assim,
        validity_flags=validity_flags,
    )
