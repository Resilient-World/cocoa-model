"""
Agrometeorology helpers shared across ingest pipelines.

This module centralises FAO-56 reference ET0 and Magnus vapor-pressure helpers so
ERA5 and CMIP6 ingest can compute consistent derived variables.
"""

from __future__ import annotations

import math

import ee
import numpy as np

# Kelvin → Celsius offset
KELVIN_OFFSET = 273.15

# Magnus (Alduchov & Eskridge 1996) — saturation vapor pressure [kPa]
MAGNUS_A = 0.61094
MAGNUS_B = 17.625
MAGNUS_C = 243.04

# FAO-56 reference grass
FAO_ALBEDO = 0.23
FAO_GAMMA = 0.067  # kPa/°C (psychrometric constant, sea-level approx.)
WIND10_TO_WIND2_FACTOR = 4.87 / np.log(67.8 * 10.0 - 5.42)


def saturation_vapor_pressure_kpa(tmean_c: float) -> float:
    """Magnus saturation vapor pressure (kPa) at ``tmean_c`` (°C)."""
    return MAGNUS_A * math.exp(MAGNUS_B * tmean_c / (MAGNUS_C + tmean_c))


def vpd_kpa(tmean_c: float, rh_pct: float) -> float:
    """Vapor pressure deficit (kPa) from mean temperature and RH (%)."""
    rh = max(0.0, min(100.0, float(rh_pct)))
    es = saturation_vapor_pressure_kpa(float(tmean_c))
    return es * (1.0 - rh / 100.0)


def magnus_es_kpa(temp_c: ee.Image) -> ee.Image:
    """Saturation vapor pressure (kPa) from temperature (°C)."""
    return ee.Image(MAGNUS_A).multiply(temp_c.multiply(MAGNUS_B).divide(temp_c.add(MAGNUS_C)).exp())


def fao_et0_daily(
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
    es = magnus_es_kpa(tmean_c)
    ea = es.multiply(rh_pct.divide(100.0))
    vpd = es.subtract(ea).max(0)

    delta = es.multiply(MAGNUS_B).multiply(MAGNUS_C).divide(tmean_c.add(MAGNUS_C).pow(2))

    u2 = wind10m.multiply(WIND10_TO_WIND2_FACTOR)
    rn = srad_mj.multiply(1.0 - FAO_ALBEDO)

    t_k = tmean_c.add(KELVIN_OFFSET)
    num_rad = delta.multiply(rn).multiply(0.408)
    num_aero = ee.Image(FAO_GAMMA).multiply(ee.Image(900).divide(t_k)).multiply(u2).multiply(vpd)
    den = delta.add(ee.Image(FAO_GAMMA).multiply(ee.Image(1).add(u2.multiply(0.34))))

    return num_rad.add(num_aero).divide(den).max(0).rename("et0")
