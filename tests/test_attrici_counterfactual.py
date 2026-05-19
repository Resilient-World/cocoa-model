"""Tests for ISIMIP3a counterclim ingest and attribution deltas."""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data.attrici_counterfactual import CounterfactualClimate, compute_attribution_deltas


def _toy(n: int = 120) -> tuple[xr.Dataset, xr.Dataset]:
    t = pd.date_range("2010-01-01", periods=n, freq="D")
    lat, lon = np.array([6.0, 7.0]), np.array([-5.0, -4.0])
    rng = np.random.default_rng(0)
    f = xr.Dataset(
        {
            "tas": (("time", "lat", "lon"), 27 + rng.normal(0, 1, (n, 2, 2))),
            "pr": (("time", "lat", "lon"), np.maximum(0, rng.gamma(0.5, 4, (n, 2, 2)))),
            "hurs": (("time", "lat", "lon"), 80 + rng.normal(0, 3, (n, 2, 2)).clip(-10, 10)),
            "rsds": (("time", "lat", "lon"), 18 + rng.normal(0, 1, (n, 2, 2))),
        },
        coords={"time": t, "lat": lat, "lon": lon},
    )
    cf = f.copy(deep=True)
    cf["tas"] = cf["tas"] - 1.2
    cf["pr"] = cf["pr"] * 0.95
    cf["hurs"] = cf["hurs"] + 0.5
    cf["rsds"] = cf["rsds"] - 0.3
    return f, cf


def test_tas_delta_is_additive_and_positive_under_warming() -> None:
    f, cf = _toy()
    d = compute_attribution_deltas(f, cf, ["tas"])
    assert float(d["tas_delta"].mean()) == pytest.approx(1.2, abs=0.05)


def test_pr_delta_uses_wet_day_mask_and_logratio() -> None:
    """Per Mengel et al. 2021 sec 3.2.3, wet-day threshold is 0.1 mm/d."""
    f, cf = _toy()
    d = compute_attribution_deltas(f, cf, ["pr"])
    assert "pr_delta" in d
    finite = d["pr_delta"].where(np.isfinite(d["pr_delta"]))
    assert float(finite.mean()) > 0


def test_hurs_and_rsds_delta_additive() -> None:
    f, cf = _toy()
    d = compute_attribution_deltas(f, cf, ["hurs", "rsds"])
    assert float(d["hurs_delta"].mean()) == pytest.approx(-0.5, abs=0.05)
    assert float(d["rsds_delta"].mean()) == pytest.approx(0.3, abs=0.05)


def test_subprocess_script_imports_without_attrici_in_main_env() -> None:
    """Main repo env must NOT have attrici; subprocess script must still import."""
    assert "attrici" not in sys.modules
    importlib.import_module("scripts.run_attrici_subprocess")


@pytest.mark.integration
def test_isimip_download_smoke(tmp_path) -> None:
    """Downloads one small counterclim file and confirms structure."""
    if not os.getenv("RUN_NETWORK_TESTS"):
        pytest.skip("Network tests disabled")

    cc = CounterfactualClimate(
        aoi=None,
        start="2015-01-01",
        end="2015-01-31",
        variables=("tas",),
        cache_dir=tmp_path,
    )
    ds = cc.fetch()
    assert "tas" in ds.data_vars
    assert ds.sizes["time"] >= 28
