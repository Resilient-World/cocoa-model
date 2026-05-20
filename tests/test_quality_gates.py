"""Tests for CI quality-gate validators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.run_evaluation import run_causal_evaluation
from analysis.validate_parallel_trends import validate_parallel_trends_report
from analysis.validate_smd import validate_smd_report
from analysis.run_evaluation import _synthetic_panel
from models.conformal import SplitConformalYield, save_conformal
from models.validate_conformal_coverage import validate_conformal_coverage


def test_validate_smd_pass_and_fail() -> None:
    validate_smd_report({"max_smd": 0.05})
    with pytest.raises(SystemExit):
        validate_smd_report({"max_smd": 0.15})


def test_validate_parallel_trends_pass_and_fail() -> None:
    validate_parallel_trends_report({"pretrend_pvalue": 0.25})
    with pytest.raises(SystemExit):
        validate_parallel_trends_report({"pretrend_pvalue": 0.01})


def test_validate_conformal_coverage_pass_and_fail(tmp_path: Path) -> None:
    path = tmp_path / "conformal.json"
    save_conformal(
        SplitConformalYield(
            quantile=1.0,
            alpha=0.1,
            validation={"empirical_coverage": 0.91, "nominal_coverage": 0.9},
        ),
        path,
    )
    with path.open() as handle:
        payload = json.load(handle)
    validate_conformal_coverage(payload)

    payload["validation"]["empirical_coverage"] = 0.80
    with pytest.raises(SystemExit):
        validate_conformal_coverage(payload)


def test_run_evaluation_synthetic_report() -> None:
    report = run_causal_evaluation(_synthetic_panel(400, seed=0))
    assert "max_smd" in report
    assert "pretrend_pvalue" in report
    assert report["max_smd"] < 0.10


def test_validate_smd_cli(tmp_path: Path) -> None:
    from analysis.validate_smd import main

    path = tmp_path / "report.json"
    path.write_text(json.dumps({"max_smd": 0.04}))
    assert main([str(path)]) == 0


def test_validate_parallel_trends_cli(tmp_path: Path) -> None:
    from analysis.validate_parallel_trends import main

    path = tmp_path / "report.json"
    path.write_text(json.dumps({"pretrend_pvalue": 0.42}))
    assert main([str(path)]) == 0


def test_run_evaluation_cli(tmp_path: Path) -> None:
    from analysis.run_evaluation import main

    out = tmp_path / "causal_eval.json"
    assert main(["--out", str(out), "--synthetic-n", "200"]) == 0
    data = json.loads(out.read_text())
    assert "max_smd" in data


def test_validate_conformal_coverage_cli(tmp_path: Path) -> None:
    from models.validate_conformal_coverage import main

    path = tmp_path / "conformal.json"
    save_conformal(
        SplitConformalYield(
            quantile=1.0,
            validation={"empirical_coverage": 0.9, "nominal_coverage": 0.9},
        ),
        path,
    )
    assert main([str(path)]) == 0
