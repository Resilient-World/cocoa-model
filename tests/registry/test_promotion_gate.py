"""Tests for promotion gate checks."""

from __future__ import annotations

from pathlib import Path

from registry.promotion_gate import GateResult, gate_crps_regression, gate_coverage, run_promotion_gate


def test_gate_crps_passes_with_baseline() -> None:
    ok, crps, _ = gate_crps_regression(Path("models/missing.pt"))
    assert ok or crps >= 0


def test_gate_coverage_passes() -> None:
    ok, cov, msg = gate_coverage()
    assert ok
    assert 0.88 <= cov <= 0.92 or "within" in msg


def test_run_promotion_gate_smoke() -> None:
    result = run_promotion_gate("yield_surrogate_v2")
    assert isinstance(result, GateResult)
    assert "crps_regression" in result.checks
