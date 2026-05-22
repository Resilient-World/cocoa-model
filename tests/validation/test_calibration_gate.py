"""Calibration gate with fixture baseline."""

from __future__ import annotations

import json
from pathlib import Path

from validation.calibration_metrics import CalibrationReport, run_calibration_gate

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "promotion" / "baseline_calibration.json"
)


def test_gate_against_fixture() -> None:
    baseline = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    report = CalibrationReport(
        model="cqr_yield",
        nominal_coverage=0.9,
        empirical_coverage=0.905,
        crps=0.11,
        ece=0.025,
        sharpness=0.44,
        pit_chi2_p=0.12,
        pit_shape="uniform",
        crpss_climatology=0.06,
        crpss_persistence=0.07,
        crpss_fdp_mean=0.03,
    )
    ok, _ = run_calibration_gate(report, baseline)
    assert ok
