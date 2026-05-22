"""Tests for api.financial multi-currency valuation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import finance.pricing as pricing
from api.financial import calculate_financial_impact


@pytest.fixture(autouse=True)
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "finance"
    cache.mkdir()
    monkeypatch.setattr(pricing, "CACHE_DIR", cache)
    monkeypatch.setattr(pricing, "ICCO_CACHE", cache / "icco_daily.parquet")
    monkeypatch.setattr(pricing, "FX_GHS_CACHE", cache / "fx_usd_ghs_daily.parquet")
    monkeypatch.setattr(pricing, "FX_XOF_CACHE", cache / "fx_usd_xof_daily.parquet")
    monkeypatch.setattr(pricing, "FUTURES_CACHE", cache / "ice_cocoa_futures.parquet")
    dates = pd.date_range("2024-06-01", periods=5, freq="D")
    pd.DataFrame({"date": dates, "icco_ny_usd_per_tonne": 10_000.0}).to_parquet(
        pricing.ICCO_CACHE, index=False
    )
    pd.DataFrame({"date": dates, "usd_per_ghs": 12.0}).to_parquet(pricing.FX_GHS_CACHE, index=False)
    pd.DataFrame({"date": dates, "usd_per_xof": 600.0}).to_parquet(
        pricing.FX_XOF_CACHE, index=False
    )


def test_calculate_financial_impact_tri_currency() -> None:
    fin = calculate_financial_impact(
        10.0,
        currency="GHS",
        pricing_basis="spot",
        farm_gate=True,
        country_code="CIV",
        cocoa_price_usd=3200.0,
        ci_low_tonnes=8.0,
        ci_high_tonnes=12.0,
    )
    assert fin.usd.point == pytest.approx(10.0 * 3200.0 * 0.65)
    assert fin.ghs.point == pytest.approx(fin.usd.point * 12.0)
    assert fin.xof.point == pytest.approx(fin.usd.point * 600.0)
    assert fin.primary.currency == "GHS"
    assert fin.primary.point == fin.ghs.point
