"""Tests for finance.pricing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import finance.pricing as pricing


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "finance"
    cache.mkdir()
    monkeypatch.setattr(pricing, "CACHE_DIR", cache)
    monkeypatch.setattr(pricing, "ICCO_CACHE", cache / "icco_daily.parquet")
    monkeypatch.setattr(pricing, "FX_GHS_CACHE", cache / "fx_usd_ghs_daily.parquet")
    monkeypatch.setattr(pricing, "FX_XOF_CACHE", cache / "fx_usd_xof_daily.parquet")
    monkeypatch.setattr(pricing, "FUTURES_CACHE", cache / "ice_cocoa_futures.parquet")


def test_farm_gate_pass_through_factors() -> None:
    ny = 10_000.0
    assert pricing.farm_gate_price_usd(ny, "GHA") == pytest.approx(7200.0)
    assert pricing.farm_gate_price_usd(ny, "CIV") == pytest.approx(6500.0)
    assert pricing.farm_gate_price_usd(ny, "CMR") == pytest.approx(8100.0)


def test_resolve_price_spot_farm_gate() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    pd.DataFrame({"date": dates, "icco_ny_usd_per_tonne": 8000.0}).to_parquet(
        pricing.ICCO_CACHE, index=False
    )
    p = pricing.resolve_price_usd_per_tonne(
        pricing_basis="spot",
        farm_gate=True,
        country_code="GHA",
    )
    assert p == pytest.approx(8000.0 * 0.72)


def test_forward_curve_tenors() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    pd.DataFrame({"date": dates, "icco_ny_usd_per_tonne": 5000.0}).to_parquet(
        pricing.ICCO_CACHE, index=False
    )
    curve = pricing.fetch_forward_curve()
    assert set(curve["tenor_months"].tolist()) == {3, 6, 12}


def test_infer_country_ghana() -> None:
    assert pricing.infer_country_code(6.5, -1.2) == "GHA"
