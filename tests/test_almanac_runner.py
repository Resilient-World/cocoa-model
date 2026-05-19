"""Tests for ALMANAC subprocess runner (stdlib only, no R)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.almanac_runner import (
    ALMANACNotInstalled,
    ALMANACRunner,
    REQUIRED_WEATHER_COLUMNS,
    _parse_dssat_table,
    _write_mgt,
    _write_sol,
    _write_wth,
)


def _toy_weather(n: int = 3000) -> pd.DataFrame:
    t = pd.date_range("2010-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "date": t,
            "tmin_c": 22 + rng.normal(0, 1, n),
            "tmax_c": 30 + rng.normal(0, 1, n),
            "precip_mm": np.maximum(0, rng.gamma(1, 5, n)),
            "srad_mj": 18 + rng.normal(0, 1, n),
            "vapor_pressure_kpa": 2.0 + rng.normal(0, 0.1, n),
        }
    )


def test_almanac_runner_imports() -> None:
    from models.almanac_runner import ALMANACResult  # noqa: F401

    assert ALMANACRunner is not None


def test_wth_sol_mgt_roundtrip(tmp_path: Path) -> None:
    weather = _toy_weather(400)
    wth = tmp_path / "COCO.WTH"
    sol = tmp_path / "COCO.SOL"
    mgt = tmp_path / "COCO.MGT"
    _write_wth(wth, weather, station="COCO", lat=6.0, lon=-2.0, elev=200.0)
    _write_sol(sol, {}, station="COCO", lat=6.0, lon=-2.0)
    _write_mgt(mgt, {"planting_density": 1100}, station="COCO", n_years=2)
    assert wth.read_text().startswith("*WEATHER")
    assert "@DATE" in wth.read_text()
    assert sol.read_text().startswith("*SOIL")
    assert mgt.read_text().startswith("*MANAGEMENT")


def test_parse_dssat_table_pln_style(tmp_path: Path) -> None:
    sample = tmp_path / "TEST.PLN"
    sample.write_text(
        "*PLANT OUTPUT\n\n@DATE LAI SW1 SW2\n"
        "10001 2.1 0.25 0.30\n"
        "10002 2.3 0.24 0.29\n",
        encoding="ascii",
    )
    df = _parse_dssat_table(sample)
    assert list(df.columns) == ["DATE", "LAI", "SW1", "SW2"]
    assert len(df) == 2


def test_init_raises_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALMANAC_BINARY", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(ALMANACNotInstalled):
        ALMANACRunner()


@pytest.mark.integration
def test_almanac_subprocess_smoke(tmp_path: Path) -> None:
    if shutil.which("almanac") is None:
        pytest.skip("ALMANAC binary not on PATH")

    runner = ALMANACRunner()
    weather = _toy_weather(8 * 365)
    soil = {"layer_depths_cm": [30, 60, 60]}
    management = {
        "station_id": "COCO",
        "latitude": 6.0,
        "longitude": -2.0,
        "planting_density": 1100,
    }
    result = runner.simulate(weather, soil, management, n_years=8)
    assert result.yearly_yield_kg_ha.size >= 1
    assert result.daily_lai.size >= 365
    assert result.daily_swc.ndim >= 1
    assert all(c in weather.columns for c in REQUIRED_WEATHER_COLUMNS)
