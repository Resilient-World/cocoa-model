"""Tests for external validation benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from validation.cocoa_barometer_check import run_barometer_check
from validation.giews_drought_validation import run_giews_validation
from validation.icco_yield_backtest import regression_metrics, run_icco_backtest
from validation.kalischek_benchmark import segmentation_metrics
from validation.kalischek_benchmark import (
    HeuristicKalischekReference,
    run_kalischek_benchmark,
    spatial_holdout_mask,
)
from validation.run_validate import run_all


class CorrelatedPredictor:
    def __init__(self, ref: HeuristicKalischekReference) -> None:
        self.ref = ref

    def sample_predictions(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        r = self.ref.sample_reference(lats, lons)
        return np.clip(r + 0.02, 0.0, 1.0)


def test_segmentation_metrics_perfect() -> None:
    y = np.array([1, 0, 1, 1], dtype=bool)
    m = segmentation_metrics(y, y)
    assert m["iou"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)


def test_spatial_holdout_fraction() -> None:
    lats = np.linspace(5, 8, 500)
    lons = np.linspace(-6, -3, 500)
    mask = spatial_holdout_mask(lats, lons, fraction=0.1, seed=0)
    assert 0.05 <= mask.mean() <= 0.25


def test_kalischek_benchmark_passes_with_correlated_predictor() -> None:
    ref = HeuristicKalischekReference()
    result = run_kalischek_benchmark(
        reference=ref,
        predictor=CorrelatedPredictor(ref),
        n_samples_per_region=500,
    )
    assert result.metrics["iou"] >= 0.55
    assert result.passed


def test_icco_backtest_mape_gate() -> None:
    result = run_icco_backtest()
    assert result.metrics["mape"] <= 0.25
    assert result.passed


def test_regression_metrics() -> None:
    obs = np.array([100.0, 110.0, 90.0])
    pred = np.array([102.0, 108.0, 92.0])
    m = regression_metrics(obs, pred)
    assert m["mape"] < 0.05


def test_barometer_and_giews_pass() -> None:
    bar = run_barometer_check()
    giews = run_giews_validation()
    assert bar.passed
    assert giews.passed


def test_run_all_writes_reports(tmp_path: Path) -> None:
    results = run_all(reports_dir=tmp_path, fail_fast=True)
    assert len(results) == 4
    assert (tmp_path / "summary.md").is_file()
    assert (tmp_path / "kalischek_benchmark.md").is_file()


def test_kalischek_cli_fail_on_low_iou(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BadPredictor:
        def sample_predictions(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
            return np.zeros(lats.size)

    from validation import kalischek_benchmark as kb
    from validation._report import ValidationResult

    monkeypatch.setattr(
        kb,
        "run_kalischek_benchmark",
        lambda **kw: ValidationResult(
            name="t",
            passed=False,
            metrics={"iou": 0.1},
            gate_description="test",
            notes=[],
        ),
    )
    assert kb.main(["--report", str(tmp_path / "k.md")]) == 1
