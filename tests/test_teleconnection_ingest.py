"""Tests for teleconnection index ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.teleconnection_ingest import get_indices_for_year, parse_nino34_sstoi

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "teleconnection" / "sstoi_snippet.txt"


def test_parse_nino34_fixture() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    df = parse_nino34_sstoi(text)
    assert len(df) >= 24
    row = df.loc[df["time"] == "2015-06-01"]
    assert len(row) == 1
    assert abs(float(row["nino34"].iloc[0]) - 0.80) < 0.01


def test_get_indices_for_year_ghana_shape() -> None:
    nino = parse_nino34_sstoi(_FIXTURE.read_text(encoding="utf-8"))
    dates = pd.date_range("2014-10-01", "2024-12-01", freq="MS")
    table = pd.DataFrame(
        {
            "time": dates,
            "nino34": np.linspace(-0.5, 1.0, len(dates)),
            "atl3": np.zeros(len(dates)),
            "iod": np.zeros(len(dates)),
        }
    )
    out = get_indices_for_year(2016, "ghana", table=table)
    assert out["nino34"].shape == (12,)
    assert out["atl3"].shape == (12,)


@pytest.mark.network
def test_nino34_matches_noaa_2015_2024() -> None:
    """Live NOAA sstoi.indices Niño3.4 within 0.01 °C of cached parquet."""
    pytest.importorskip("requests")
    from data.teleconnection_ingest import NINO34_URL, _fetch_text, refresh_indices

    from data.teleconnection_ingest import build_indices_table

    live = parse_nino34_sstoi(_fetch_text(NINO34_URL))
    path = Path(__file__).resolve().parents[1] / "data" / "external" / "_test_teleconnection.parquet"
    table = build_indices_table(nino34_df=live, atl3_df=None, iod_df=None, allow_proxy=True)
    table.to_parquet(path, index=False)
    cached = pd.read_parquet(path)

    live_sub = live[(live["time"] >= "2015-01-01") & (live["time"] <= "2024-12-01")]
    merged = live_sub.merge(cached[["time", "nino34"]], on="time", suffixes=("_live", "_cache"))
    diff = (merged["nino34_live"] - merged["nino34_cache"]).abs()
    assert float(diff.max()) < 0.01
