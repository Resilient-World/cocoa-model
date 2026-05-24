from __future__ import annotations

import pytest

from counterfactual.cmip7_scenarios import CMIP7ScenarioBuilder
from counterfactual.cmip_factory import build_cmip_scenario_builder


def test_cmip7_factory_returns_clear_placeholder_warning(tmp_path) -> None:
    builder = build_cmip_scenario_builder(
        historical_zarr_path=tmp_path / "hist.zarr",
        cmip6_zarr_path=tmp_path / "cmip6.zarr",
        cmip7_zarr_path=tmp_path / "cmip7.zarr",
        version="cmip7",
    )
    assert isinstance(builder, CMIP7ScenarioBuilder)
    with pytest.raises(FileNotFoundError, match="CMIP7 ensemble not yet published"):
        builder.build_scenario("SSP2-4.5", ("2050-01-01", "2050-12-31"))
