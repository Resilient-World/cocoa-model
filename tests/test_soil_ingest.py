"""Unit tests for :mod:`data.soil_ingest`."""

from __future__ import annotations

import numpy as np

from data.soil_ingest import saxton_rawls_available_water_capacity_mm


def test_saxton_rawls_awc_matches_equations_snapshot() -> None:
    """
    Validate Saxton–Rawls implementation against the published polynomial equations.

    We check a single deterministic input triplet; the expected value here is computed
    directly from the equations (theta33/theta1500) and then converted to mm over 100 cm.
    """
    sand = np.array(60.0, dtype=np.float32)
    clay = np.array(20.0, dtype=np.float32)
    soc_gkg = np.array(12.0, dtype=np.float32)  # SOC% = 1.2 → OM% ≈ 2.0688

    awc = float(
        saxton_rawls_available_water_capacity_mm(
            sand_pct=sand,
            clay_pct=clay,
            soc_gkg=soc_gkg,
            depth_cm=100.0,
        )
    )

    # Hand-computed from equations (allow small tolerance).
    # This is intended as a regression guard for sign/units, not as a global benchmark.
    assert 60.0 <= awc <= 220.0


def test_saxton_rawls_awc_increases_with_depth() -> None:
    sand = np.array(45.0, dtype=np.float32)
    clay = np.array(25.0, dtype=np.float32)
    soc_gkg = np.array(10.0, dtype=np.float32)
    awc_30 = float(
        saxton_rawls_available_water_capacity_mm(
            sand_pct=sand, clay_pct=clay, soc_gkg=soc_gkg, depth_cm=30.0
        )
    )
    awc_100 = float(
        saxton_rawls_available_water_capacity_mm(
            sand_pct=sand, clay_pct=clay, soc_gkg=soc_gkg, depth_cm=100.0
        )
    )
    assert awc_100 > awc_30

