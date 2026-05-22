"""
CASEJ cocoa process-model emulator (Asante et al. 2025).

Implements daily water-limited growth with CO2 fertilization, cumulative heat stress
above 32 °C, and shade-LAI moderation of VPD. Used to generate PINN training targets
when the RCASEJ / CASEJ Fortran engine is not installed locally.
"""

from __future__ import annotations

import structlog

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARAMS_PATH = _REPO_ROOT / "config" / "casej" / "params.yaml"
GRAMS_PER_TONNE = 1_000_000.0


@dataclass(frozen=True)
class CASEJParams:
    co2_ppm_min: float = 380.0
    co2_ppm_max: float = 700.0
    co2_ref_ppm: float = 400.0
    co2_f_max: float = 1.45
    co2_beta_ln: float = 0.38
    t_opt_c: float = 27.0
    t_base_c: float = 18.0
    t_max_c: float = 40.0
    heat_threshold_c: float = 32.0
    heat_decline_per_cdd: float = 0.0035
    kc: float = 1.05
    rue_mj: float = 1.65
    harvest_index: float = 0.12
    awc_default_mm: float = 150.0
    k_vpd_moderation: float = 0.35
    slai_ref: float = 2.0
    slai_max: float = 3.5
    tree_age_peak_y: float = 15.0
    tree_age_ramp_y: float = 5.0
    tree_age_senescence_y: float = 30.0


def load_casej_params(path: Path | None = None) -> CASEJParams:
    """Load parameters from ``config/casej/params.yaml``."""
    cfg_path = path or DEFAULT_PARAMS_PATH
    with cfg_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    co2 = raw.get("co2", {})
    temp = raw.get("temperature", {})
    water = raw.get("water", {})
    shade = raw.get("shade", {})
    phen = raw.get("phenology", {})
    return CASEJParams(
        co2_ppm_min=float(co2.get("ppm_min", 380)),
        co2_ppm_max=float(co2.get("ppm_max", 700)),
        co2_ref_ppm=float(co2.get("ref_ppm", 400)),
        co2_f_max=float(co2.get("f_max", 1.45)),
        co2_beta_ln=float(co2.get("beta_ln", 0.38)),
        t_opt_c=float(temp.get("t_opt_c", 27)),
        t_base_c=float(temp.get("t_base_c", 18)),
        t_max_c=float(temp.get("t_max_c", 40)),
        heat_threshold_c=float(temp.get("heat_threshold_c", 32)),
        heat_decline_per_cdd=float(temp.get("heat_decline_per_cdd", 0.0035)),
        kc=float(water.get("kc", 1.05)),
        rue_mj=float(water.get("rue_mj_per_g", 1.65)),
        harvest_index=float(water.get("harvest_index", 0.12)),
        awc_default_mm=float(water.get("awc_default_mm", 150)),
        k_vpd_moderation=float(shade.get("k_vpd_moderation", 0.35)),
        slai_ref=float(shade.get("slai_ref", 2.0)),
        slai_max=float(shade.get("slai_max", 3.5)),
        tree_age_peak_y=float(phen.get("tree_age_peak_y", 15)),
        tree_age_ramp_y=float(phen.get("tree_age_ramp_y", 5)),
        tree_age_senescence_y=float(phen.get("tree_age_senescence_y", 30)),
    )


def co2_fertilization_factor(co2_ppm: float | np.ndarray, params: CASEJParams) -> np.ndarray:
    """Monotonic CO2 response in 380–700 ppm (log-scale, capped)."""
    co2 = np.clip(np.asarray(co2_ppm, dtype=np.float64), params.co2_ppm_min, params.co2_ppm_max)
    raw = 1.0 + params.co2_beta_ln * np.log(co2 / params.co2_ref_ppm)
    return np.clip(raw, 1.0, params.co2_f_max)


def temperature_stress(tmean_c: np.ndarray, params: CASEJParams) -> np.ndarray:
    rise = (tmean_c - params.t_base_c) / (params.t_opt_c - params.t_base_c)
    fall = (params.t_max_c - tmean_c) / (params.t_max_c - params.t_opt_c)
    return np.clip(np.minimum(rise, fall), 0.0, 1.0)


def heat_stress_factor(tmax_c: np.ndarray, params: CASEJParams) -> float:
    """Decline from cumulative degree-days above ``heat_threshold_c``."""
    cdd = float(np.sum(np.maximum(tmax_c - params.heat_threshold_c, 0.0)))
    return float(np.clip(1.0 - params.heat_decline_per_cdd * cdd, 0.05, 1.0))


def shade_vpd_factor(vpd_kpa: np.ndarray, slai: float, params: CASEJParams) -> np.ndarray:
    """Higher shade LAI moderates VPD stress (agroforestry)."""
    slai_norm = float(np.clip(slai / params.slai_max, 0.0, 1.0))
    moderation = 1.0 - params.k_vpd_moderation * slai_norm
    excess = np.maximum(vpd_kpa - 1.65, 0.0)
    return np.clip(1.0 - moderation * excess / 2.0, 0.1, 1.0)


def age_factor(tree_age_y: float, params: CASEJParams) -> float:
    age = float(tree_age_y)
    if age < params.tree_age_ramp_y:
        return float(age / params.tree_age_ramp_y)
    if age <= params.tree_age_peak_y:
        return 1.0
    if age < params.tree_age_senescence_y:
        return float(
            1.0
            - 0.4
            * (age - params.tree_age_peak_y)
            / (params.tree_age_senescence_y - params.tree_age_peak_y)
        )
    return 0.6


def water_yield_cap(annual_precip_mm: float, annual_et_mm: float) -> float:
    """Upper bound from annual water balance (mm → t/ha scale factor)."""
    if annual_et_mm <= 1.0:
        return 3.5
    ratio = annual_precip_mm / annual_et_mm
    return float(np.clip(0.35 + 1.8 * ratio, 0.2, 3.5))


def co2_ppm_for_ssp(scenario: str, horizon_year: int, params_path: Path | None = None) -> float:
    """Interpolate SSP CO2 levels from ``params.yaml``."""
    with (params_path or DEFAULT_PARAMS_PATH).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    table: dict[str, dict[str, float]] = raw.get("ssp_co2_ppm", {})
    key = scenario.strip().lower()
    if not key.startswith("ssp"):
        key = f"ssp{key}"
    years_map = table.get(key) or table.get("ssp245", {"2023": 415, "2050": 520, "2100": 600})

    def _ppm(year: int) -> float:
        if str(year) in years_map:
            return float(years_map[str(year)])
        if year in years_map:
            return float(years_map[year])
        raise KeyError(year)

    years = sorted(int(y) for y in years_map)
    if horizon_year <= years[0]:
        return _ppm(years[0])
    if horizon_year >= years[-1]:
        return _ppm(years[-1])
    for y0, y1 in zip(years[:-1], years[1:], strict=True):
        if y0 <= horizon_year <= y1:
            c0 = _ppm(y0)
            c1 = _ppm(y1)
            w = (horizon_year - y0) / (y1 - y0)
            return c0 + w * (c1 - c0)
    return 520.0


@dataclass
class CASEJSite:
    lat: float
    lon: float
    awc_mm: float
    slai: float
    tree_age_y: float
    co2_ppm: float


def synthesize_daily_weather(
    n_days: int,
    *,
    seed: int,
    temp_offset_c: float = 0.0,
    precip_scale: float = 1.0,
) -> pd.DataFrame:
    """West-Africa-like synthetic daily weather for LHS training."""
    rng = np.random.default_rng(seed)
    day = np.arange(n_days, dtype=np.float64)
    seasonal = np.sin(2 * np.pi * day / 365.0)
    tmax = 31.0 + 2.5 * seasonal + temp_offset_c + rng.normal(0, 0.3, n_days)
    tmin = tmax - 7.0 + rng.normal(0, 0.2, n_days)
    precip = np.clip(rng.gamma(2.0, 4.0, n_days) * precip_scale, 0.0, 80.0)
    srad = np.clip(12.0 + 4.0 * seasonal + rng.normal(0, 0.5, n_days), 4.0, 22.0)
    vpd = np.clip(0.8 + 0.6 * (1.0 - seasonal) + rng.normal(0, 0.1, n_days), 0.2, 2.5)
    et0 = np.clip(2.5 + 0.8 * (1.0 - seasonal), 1.0, 6.0)
    return pd.DataFrame(
        {
            "tmax_c": tmax.astype(np.float32),
            "tmin_c": tmin.astype(np.float32),
            "tmean_c": (0.5 * (tmax + tmin)).astype(np.float32),
            "precip_mm": precip.astype(np.float32),
            "srad_mj": srad.astype(np.float32),
            "vpd_kpa": vpd.astype(np.float32),
            "et0_mm": et0.astype(np.float32),
        }
    )


def run_casej_yearly(
    weather: pd.DataFrame,
    site: CASEJSite,
    params: CASEJParams | None = None,
) -> dict[str, float]:
    """
    Run one water-limited CASEJ year → yield (t/ha) and diagnostic scalars.
    """
    p = params or load_casej_params()
    tmean = weather["tmean_c"].to_numpy(dtype=np.float64)
    tmax = weather["tmax_c"].to_numpy(dtype=np.float64)
    precip = weather["precip_mm"].to_numpy(dtype=np.float64)
    srad = weather["srad_mj"].to_numpy(dtype=np.float64)
    vpd = weather["vpd_kpa"].to_numpy(dtype=np.float64)
    et0 = weather["et0_mm"].to_numpy(dtype=np.float64)

    f_co2 = float(co2_fertilization_factor(site.co2_ppm, p))
    f_temp = temperature_stress(tmean, p)
    f_heat = heat_stress_factor(tmax, p)
    f_vpd = shade_vpd_factor(vpd, site.slai, p)
    f_age = age_factor(site.tree_age_y, p)

    awc = max(site.awc_mm, 50.0)
    sw = 0.5 * awc
    biomass_g = 0.0
    for t in range(len(tmean)):
        f_w = float(np.clip(sw / (0.45 * awc), 0.0, 1.0))
        d_b = (
            p.rue_mj
            * srad[t]
            * f_w
            * f_vpd[t]
            * f_temp[t]
            * f_co2
            * f_age
        )
        biomass_g += d_b
        et_crop = et0[t] * p.kc
        sw = float(np.clip(sw + precip[t] - et_crop, 0.0, awc))

    y_raw = p.harvest_index * biomass_g / GRAMS_PER_TONNE
    annual_ppt = float(precip.sum())
    annual_et = float((et0 * p.kc).sum())
    y_cap = water_yield_cap(annual_ppt, annual_et)
    y = float(np.clip(min(y_raw, y_cap) * f_heat, 0.0, 3.5))
    return {
        "yield_t_ha": y,
        "f_co2": f_co2,
        "f_heat": f_heat,
        "heat_cdd": float(np.sum(np.maximum(tmax - p.heat_threshold_c, 0.0))),
        "annual_precip_mm": annual_ppt,
        "annual_et_mm": annual_et,
        "water_cap_t_ha": y_cap,
    }


def weather_to_climate_tensor(weather: pd.DataFrame, co2_ppm: float) -> np.ndarray:
    """Pack daily weather into ``[365, 11]`` surrogate channel order."""
    from models.surrogate.yield_surrogate import CLIMATE_IDX, N_CLIMATE_CHANNELS

    n = len(weather)
    out = np.zeros((n, N_CLIMATE_CHANNELS), dtype=np.float32)
    out[:, CLIMATE_IDX["tmax"]] = weather["tmax_c"].to_numpy()
    out[:, CLIMATE_IDX["tmin"]] = weather["tmin_c"].to_numpy()
    out[:, CLIMATE_IDX["tmean"]] = weather["tmean_c"].to_numpy()
    out[:, CLIMATE_IDX["precip"]] = weather["precip_mm"].to_numpy()
    out[:, CLIMATE_IDX["srad"]] = weather["srad_mj"].to_numpy()
    out[:, CLIMATE_IDX["vpd"]] = weather["vpd_kpa"].to_numpy()
    out[:, CLIMATE_IDX["et0"]] = weather["et0_mm"].to_numpy()
    out[:, CLIMATE_IDX["sm_root"]] = 0.28
    out[:, CLIMATE_IDX["wind10m"]] = 2.0
    out[:, CLIMATE_IDX["rh_mean"]] = 75.0
    out[:, CLIMATE_IDX["co2_ppm"]] = float(co2_ppm)
    return out


def site_to_static_vector(site: CASEJSite) -> np.ndarray:
    """Pack site into 13-d static vector (yield_surrogate layout)."""
    from models.surrogate.yield_surrogate import (
        cohort_phase_from_age,
        planting_density_norm,
        tree_age_years_norm,
    )

    static = np.zeros(13, dtype=np.float32)
    static[0] = site.awc_mm
    static[1] = 0.4
    static[5] = 0.3
    static[10] = tree_age_years_norm(site.tree_age_y)
    static[11] = cohort_phase_from_age(site.tree_age_y)
    static[12] = planting_density_norm(1100.0)
    static[9] = 0.7
    return static
