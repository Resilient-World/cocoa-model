"""Smoke tests for backbone benchmark script."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_sample_holdout_tiles_count() -> None:
    from scripts.benchmark_backbones import sample_holdout_tiles

    lats, lons, labels = sample_holdout_tiles(12, seed=7)
    assert len(lats) == 12
    assert labels.shape == (12, 64, 64)


def test_tile_metrics_binary() -> None:
    import numpy as np

    from scripts.benchmark_backbones import tile_metrics

    y_true = np.ones((8, 8), dtype=np.uint8)
    y_prob = np.ones((8, 8)) * 0.9
    m = tile_metrics(y_true, y_prob, threshold=0.5)
    assert m["miou"] == pytest.approx(1.0, abs=0.01)
    assert m["f1"] == pytest.approx(1.0, abs=0.01)


def test_write_benchmark_report(tmp_path: Path) -> None:
    from scripts.benchmark_backbones import BackboneResult, write_benchmark_report

    results = [
        BackboneResult("FDP-only", 0.15, 0.8, 0.9, 0.1, 1.0, 0.0, 100),
        BackboneResult("Galileo-Base + seg head", 0.12, 0.7, 0.85, 0.12, 50.0, 90.0, 100),
        BackboneResult("Prithvi-EO-2.0", 0.14, 0.75, 0.88, 0.11, 5.0, 0.01, 100),
    ]
    out = write_benchmark_report(results, tmp_path / "bench.md", galileo_checkpoint_present=False)
    text = out.read_text()
    assert "Galileo" in text
    assert "Production backbone" in text


def test_tile_mean_error() -> None:
    import numpy as np

    from scripts.benchmark_backbones import tile_mean_error

    y_true = np.array([[0, 1], [1, 0]], dtype=np.float64)
    y_prob = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float64)
    assert tile_mean_error(y_true, y_prob) == pytest.approx(0.125)


def test_write_aef_benchmark_report(tmp_path: Path) -> None:
    from scripts.benchmark_backbones import BackboneResult, write_aef_benchmark_report

    results = [
        BackboneResult("AlphaEarth Foundations (AEF)", 0.08, 0.82, 0.91, 0.15, 2.0, 0.05, 100),
        BackboneResult("Galileo-Base + seg head", 0.12, 0.7, 0.85, 0.12, 50.0, 90.0, 100),
        BackboneResult("FDP-only", 0.15, 0.8, 0.9, 0.1, 1.0, 0.0, 100),
    ]
    out = write_aef_benchmark_report(
        results, tmp_path / "bench_aef.md", aef_checkpoint_present=True
    )
    text = out.read_text()
    assert "AlphaEarth" in text
    assert "Mean error" in text
    assert "0.5 × AEF" in text
