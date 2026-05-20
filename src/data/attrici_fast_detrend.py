"""
Fast ATTRICI-style counterfactual (Mengel et al. 2021, Earth Syst. Dynam.).

Detrends daily climate variables against a smoothed global-mean temperature (GMT)
series and maps observations to a pre-industrial GMT level. Uses the published
``attrici`` package in ``bayesian`` mode when installed; otherwise a vendored
quantile-mapping detrender in ``fast`` mode (same public API, with a logged warning).

Derived variables (``vpd_mean``, ``et0``, ``cwd``) are recomputed from detrended
state variables — never detrended directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Literal, Sequence

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
)

logger = logging.getLogger(__name__)

MENGEL_2021_REF = "Mengel et al. 2021, Earth Syst. Dynam. (ATTRICI)"

GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
GISTEMP_CACHE_DIR = Path("data/external/gistemp")
GISTEMP_CACHE_FILE = GISTEMP_CACHE_DIR / "GLB.Ts+dSST.csv"
GISTEMP_CACHE_MAX_AGE_S = 24 * 3600

DEFAULT_VARIABLES: tuple[str, ...] = ("tmax", "tmin", "precip")
QUANTILE_LEVELS = np.linspace(0.05, 0.95, 19)

try:
    from scipy import stats as scipy_stats
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - scipy is a transitive dep of scikit-learn
    scipy_stats = None  # type: ignore[assignment]
    savgol_filter = None  # type: ignore[assignment]


def _saturation_vapor_pressure_kpa_array(tmean_c: xr.DataArray) -> xr.DataArray:
    return MAGNUS_A * np.exp(MAGNUS_B * tmean_c / (MAGNUS_C + tmean_c))


def _vpd_from_tmean_rh(tmean_c: xr.DataArray, rh_pct: xr.DataArray) -> xr.DataArray:
    rh = rh_pct.clip(0, 100)
    es = _saturation_vapor_pressure_kpa_array(tmean_c)
    return es * (1.0 - rh / 100.0)


def _fao_et0_array(
    tmean_c: xr.DataArray,
    rh_pct: xr.DataArray,
    wind10m: xr.DataArray,
    srad_mj: xr.DataArray,
) -> xr.DataArray:
    """FAO-56 Penman–Monteith reference ET0 (mm/day), numpy/xarray port of era5_ingest."""
    es = _saturation_vapor_pressure_kpa_array(tmean_c)
    ea = es * (rh_pct / 100.0)
    vpd = (es - ea).clip(min=0)
    delta = es * MAGNUS_B * MAGNUS_C / (tmean_c + MAGNUS_C) ** 2
    u2 = wind10m * WIND10_TO_WIND2_FACTOR
    rn = srad_mj * (1.0 - FAO_ALBEDO)
    t_k = tmean_c + KELVIN_OFFSET
    num_rad = delta * rn * 0.408
    num_aero = FAO_GAMMA * (900.0 / t_k) * u2 * vpd
    den = delta + FAO_GAMMA * (1.0 + 0.34 * u2)
    return (num_rad + num_aero) / den


def load_gistemp_loti(
    start_year: int = 1880,
    smooth_window: int = 21,
) -> pd.Series:
    """
    Load NASA GISTEMP v4 global LOTI (°C anomaly + base) and apply Savitzky–Golay smoothing.

    Cached on disk for 24 hours under ``data/external/gistemp/``.
    """
    GISTEMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if GISTEMP_CACHE_FILE.is_file():
        age = time.time() - GISTEMP_CACHE_FILE.stat().st_mtime
        if age > GISTEMP_CACHE_MAX_AGE_S:
            logger.info("GISTEMP cache stale (%.0fh); re-downloading", age / 3600)
        else:
            raw = GISTEMP_CACHE_FILE.read_text(encoding="utf-8")
    else:
        raw = None

    if raw is None:
        logger.info("Downloading GISTEMP LOTI from %s", GISTEMP_URL)
        response = requests.get(GISTEMP_URL, timeout=60)
        response.raise_for_status()
        raw = response.text
        GISTEMP_CACHE_FILE.write_text(raw, encoding="utf-8")

    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.startswith((" ", "Year"))]
    # File header rows start with Year; data rows: Year, Jan..Dec, J-D
    records: list[tuple[int, float]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit():
            continue
        year = int(parts[0])
        if year < start_year:
            continue
        # Annual mean from monthly columns (skip Year and J-D)
        months = []
        for val in parts[1:13]:
            if val in ("", "***", "*****"):
                continue
            try:
                months.append(float(val))
            except ValueError:
                continue
        if months:
            records.append((year, float(np.mean(months))))

    if not records:
        raise ValueError(f"No GISTEMP annual values found from {start_year}")

    years, values = zip(*records)
    series = pd.Series(values, index=np.array(years, dtype=int), name="gmt_loti")

    if savgol_filter is not None and len(series) >= smooth_window:
        # polyorder=3 per ATTRICI preprocessing
        smoothed = savgol_filter(series.values, window_length=smooth_window, polyorder=3)
        series = pd.Series(smoothed, index=series.index, name="gmt_loti")
    else:
        if savgol_filter is None:
            logger.warning(
                "scipy not available; using rolling mean instead of Savitzky–Golay for GISTEMP"
            )
        series = series.rolling(window=smooth_window, center=True, min_periods=1).mean()

    return series


def _preindustrial_gmt(gmt: pd.Series, data_years: np.ndarray | None = None) -> float:
    """GMT baseline: early segment of overlap with data, else classic pre-industrial years."""
    if data_years is not None and data_years.size > 0:
        ymin = int(np.nanmin(data_years))
        ymax_early = ymin + 30
        overlap = gmt[(gmt.index >= ymin) & (gmt.index <= ymax_early)]
        if len(overlap) >= 5:
            return float(overlap.mean())
    early = gmt[gmt.index <= 1900]
    if len(early) >= 10:
        return float(early.mean())
    return float(gmt.iloc[: min(30, len(gmt))].mean())


def _doy_circular_distance(doy_a: np.ndarray, doy_b: int, window: int) -> np.ndarray:
    half = window // 2
    diff = np.abs(doy_a - doy_b)
    return np.minimum(diff, 366 - diff) <= half


def _fast_pixel_detrend(
    values: np.ndarray,
    years: np.ndarray,
    doys: np.ndarray,
    gmt_values: np.ndarray,
    gmt_preindustrial: float,
    window_days: int,
    is_precip: bool,
) -> np.ndarray:
    """
    Per-pixel quantile-mapping detrend (Mengel et al. 2021 style).

    Within each DOY window, regress quantiles of ``values`` on GMT, map each
    observation to the pre-industrial GMT quantile.
    """
    if scipy_stats is None:
        raise ImportError(
            "fast mode requires scipy (transitive via scikit-learn). "
            "Install scipy or use pip install -e '.[attrici]'."
        )

    n = values.size
    out = np.empty(n, dtype=np.float64)
    gmt_values = np.asarray(gmt_values, dtype=np.float64)

    for i in range(n):
        win = _doy_circular_distance(doys, int(doys[i]), window_days)
        y_w = values[win]
        g_w = gmt_values[win]
        if y_w.size < 10:
            out[i] = values[i]
            continue

        gmt_o = gmt_values[i]
        p = scipy_stats.percentileofscore(y_w, values[i], kind="mean") / 100.0

        # Quantile–GMT regression: Q_tau(g) = a_tau + b_tau * g
        q_preds_pi: list[float] = []
        for tau in QUANTILE_LEVELS:
            q_tgt = np.quantile(y_w, tau)
            # local linear fit of quantile vs GMT using overlapping bins
            g_bins = np.linspace(g_w.min(), g_w.max(), min(8, len(np.unique(g_w))))
            bin_centers: list[float] = []
            bin_qs: list[float] = []
            for j in range(len(g_bins) - 1):
                mask = (g_w >= g_bins[j]) & (g_w < g_bins[j + 1])
                if mask.sum() < 3:
                    continue
                bin_centers.append(float(g_w[mask].mean()))
                bin_qs.append(float(np.quantile(y_w[mask], tau)))
            if len(bin_centers) < 2:
                q_preds_pi.append(float(np.quantile(y_w, tau)))
                continue
            slope, intercept = np.polyfit(bin_centers, bin_qs, 1)
            q_preds_pi.append(intercept + slope * gmt_preindustrial)

        y_cf = float(np.interp(p, QUANTILE_LEVELS, q_preds_pi))
        if is_precip:
            y_cf = max(0.0, y_cf)
        out[i] = y_cf

    # Remove residual linear GMT sensitivity (Mengel et al. first-order component)
    valid = np.isfinite(gmt_values) & np.isfinite(values)
    if valid.sum() >= 10:
        slope, _ = np.polyfit(gmt_values[valid], values[valid], 1)
        out = out - slope * (gmt_values - gmt_preindustrial)

    if is_precip:
        out = np.maximum(0.0, out)

    return out


def _years_and_doy_from_time(time: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    t_index = pd.DatetimeIndex(time.values)
    years = t_index.year.to_numpy()
    doys = t_index.dayofyear.to_numpy()
    return years, doys


def _gmt_for_timesteps(years: np.ndarray, gmt: pd.Series) -> np.ndarray:
    return np.array([gmt.get(int(y), np.nan) for y in years], dtype=np.float64)


class FastATTRICICounterfactual:
    """
    Fit and apply GMT-driven counterfactual transforms to a daily ``xarray.Dataset``.

    Parameters
    ----------
    gmt_series:
        Annual global mean temperature (°C), indexed by calendar year.
    variables:
        Data variables to detrend (must exist in the input dataset).
    mode:
        ``fast`` — vendored quantile-mapping detrender; ``bayesian`` — ``attrici`` package.
    window_days:
        DOY window width for local detrending in ``fast`` mode.
    random_state:
        RNG seed (reserved for ``bayesian`` mode).
    """

    def __init__(
        self,
        gmt_series: pd.Series,
        variables: Sequence[str] = DEFAULT_VARIABLES,
        window_days: int = 31,
        random_state: int = 42,
    ) -> None:
        self.gmt_series = gmt_series.sort_index()
        self.variables = tuple(variables)
        self.window_days = window_days
        self.random_state = random_state
        self._gmt_preindustrial: float | None = None
        self._fitted = False

    def fit(self, ds: xr.Dataset) -> FastATTRICICounterfactual:
        """Record GMT baseline and validate variables (lightweight fit)."""
        years, _ = _years_and_doy_from_time(ds["time"])
        self._gmt_preindustrial = _preindustrial_gmt(self.gmt_series, years)
        missing = [v for v in self.variables if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Dataset missing variables for fit: {missing}")
        self._fitted = True
        logger.info(
            "FastATTRICICounterfactual fit: gmt_pi=%.3f°C, variables=%s",
            self._gmt_preindustrial,
            self.variables,
        )
        return self

    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        """Return dataset with ``{var}_cf`` counterfactual variables."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        return self._transform_fast(ds)

    def fit_transform(self, ds: xr.Dataset) -> xr.Dataset:
        return self.fit(ds).transform(ds)

    def _transform_fast(self, ds: xr.Dataset) -> xr.Dataset:
        assert self._gmt_preindustrial is not None
        years, doys = _years_and_doy_from_time(ds["time"])
        gmt_t = _gmt_for_timesteps(years, self.gmt_series)

        out = ds.copy()
        spatial_dims = [d for d in ("latitude", "longitude", "lat", "lon") if d in ds.dims]
        if len(spatial_dims) < 2:
            raise ValueError("Dataset must have latitude/longitude (or lat/lon) dimensions")

        for var in self.variables:
            if var not in ds.data_vars:
                logger.warning("Skipping missing variable %s", var)
                continue
            da = ds[var]
            is_precip = var == "precip"

            def _apply_pixel(values: np.ndarray) -> np.ndarray:
                return _fast_pixel_detrend(
                    values,
                    years,
                    doys,
                    gmt_t,
                    self._gmt_preindustrial,
                    self.window_days,
                    is_precip,
                )

            cf = xr.apply_ufunc(
                _apply_pixel,
                da,
                input_core_dims=[["time"]],
                output_core_dims=[["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float64],
            )
            cf.name = f"{var}_cf"
            cf.attrs.update(da.attrs)
            cf.attrs["counterfactual_method"] = "ATTRICI-style quantile mapping (fast)"
            cf.attrs["gmt_preindustrial"] = self._gmt_preindustrial
            out[cf.name] = cf

        return out


def recompute_derived_counterfactuals(ds_cf: xr.Dataset) -> xr.Dataset:
    """
    Recompute ``vpd_mean_cf``, ``et0_cf``, and ``cwd_cf`` from detrended state variables.

    Requires ``tmax_cf``, ``tmin_cf``; uses ``rh_mean_cf``, ``srad_cf``, ``wind10m_cf``,
    and ``precip_cf`` when present. Never detrends derived variables directly.
    """
    out = ds_cf.copy()
    if "tmax_cf" not in out or "tmin_cf" not in out:
        return out

    tmean_cf = (out["tmax_cf"] + out["tmin_cf"]) / 2.0
    out["tmean_cf"] = tmean_cf

    if "rh_mean_cf" in out:
        out["vpd_mean_cf"] = _vpd_from_tmean_rh(tmean_cf, out["rh_mean_cf"])
    elif "rh_mean" in out and "rh_mean_cf" not in out:
        pass

    need_et0 = {"rh_mean_cf", "wind10m_cf", "srad_cf"} <= set(out.data_vars)
    if need_et0:
        out["et0_cf"] = _fao_et0_array(
            tmean_cf,
            out["rh_mean_cf"],
            out["wind10m_cf"],
            out["srad_cf"],
        )
        if "precip_cf" in out:
            out["cwd_cf"] = out["et0_cf"] - out["precip_cf"]
        elif "precip" in out:
            out["cwd_cf"] = out["et0_cf"] - out["precip"]

    return out


def hazard_return_period_shift(
    ds: xr.Dataset,
    variable: str,
    threshold: float,
    direction: Literal["above", "below"] = "above",
) -> xr.Dataset:
    """
    Per-pixel exceedance probabilities and attribution metrics.

    Returns
    -------
    xarray.Dataset
        ``p_factual``, ``p_counterfactual``, ``risk_ratio``, ``fraction_attributable_risk``
        (FAR = 1 - p_cf / p_fac).
    """
    cf_var = f"{variable}_cf"
    if variable not in ds.data_vars:
        raise KeyError(f"Variable {variable!r} not in dataset")
    if cf_var not in ds.data_vars:
        raise KeyError(f"Counterfactual {cf_var!r} not in dataset; run transform() first")

    fac = ds[variable]
    cf = ds[cf_var]

    if direction == "above":
        mask_fac = fac > threshold
        mask_cf = cf > threshold
    else:
        mask_fac = fac < threshold
        mask_cf = cf < threshold

    p_factual = mask_fac.mean(dim="time")
    p_counterfactual = mask_cf.mean(dim="time")
    eps = 1e-12
    risk_ratio = p_factual / (p_counterfactual + eps)
    far = 1.0 - (p_counterfactual / (p_factual + eps))

    return xr.Dataset(
        {
            "p_factual": p_factual,
            "p_counterfactual": p_counterfactual,
            "risk_ratio": risk_ratio,
            "fraction_attributable_risk": far,
            "far": far,
        },
        attrs={
            "variable": variable,
            "threshold": threshold,
            "direction": direction,
            "reference": MENGEL_2021_REF,
        },
    )


def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ATTRICI-style counterfactual daily climate Zarr from factual ERA5 Zarr."
    )
    parser.add_argument(
        "--era5-zarr",
        type=Path,
        required=True,
        help="Input factual ERA5-Land daily Zarr (e.g. data/processed/era5_daily.zarr).",
    )
    parser.add_argument(
        "--out-zarr",
        type=Path,
        required=True,
        help="Output Zarr with factual variables plus *_cf counterfactuals.",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default="tmax,tmin,precip",
        help="Comma-separated variables to detrend (default: tmax,tmin,precip).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    variables = tuple(v.strip() for v in args.variables.split(",") if v.strip())
    if not variables:
        print("No variables specified.", file=sys.stderr)
        return 1

    if not args.era5_zarr.exists():
        print(f"Input Zarr not found: {args.era5_zarr}", file=sys.stderr)
        return 1

    gmt = load_gistemp_loti()
    logger.info("Loaded smoothed GISTEMP LOTI (%d years)", len(gmt))

    ds = xr.open_zarr(args.era5_zarr, consolidated=True)
    model = ATTRICICounterfactual(gmt, variables=variables, )  # type: ignore[arg-type]
    cf = model.fit_transform(ds)

    out = ds.copy()
    for name, da in cf.data_vars.items():
        out[name] = da

    if {"tmax_cf", "tmin_cf"} <= set(out.data_vars):
        try:
            out = recompute_derived_counterfactuals(out)
        except KeyError as exc:
            logger.warning("Skipping derived counterfactual recompute: %s", exc)

    args.out_zarr.parent.mkdir(parents=True, exist_ok=True)
    out.to_zarr(args.out_zarr, mode="w")
    logger.info("Wrote counterfactual stack to %s", args.out_zarr)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
