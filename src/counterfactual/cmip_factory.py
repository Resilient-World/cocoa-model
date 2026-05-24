"""Factory for CMIP6/CMIP7 scenario builders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from counterfactual.cmip6_scenarios import ScenarioBuilder as CMIP6ScenarioBuilder
from counterfactual.cmip7_scenarios import CMIP7ScenarioBuilder
from counterfactual.cmip_scenarios_base import CMIPScenarioBuilderBase

CMIPVersion = Literal["cmip6", "cmip7"]


def cmip_version_from_env(default: CMIPVersion = "cmip6") -> CMIPVersion:
    value = os.getenv("CMIP_VERSION", default).lower()
    if value not in {"cmip6", "cmip7"}:
        raise ValueError("CMIP_VERSION must be 'cmip6' or 'cmip7'")
    return value  # type: ignore[return-value]


def build_cmip_scenario_builder(
    *,
    historical_zarr_path: str | Path,
    cmip6_zarr_path: str | Path,
    cmip7_zarr_path: str | Path | None = None,
    version: CMIPVersion | None = None,
) -> CMIPScenarioBuilderBase:
    selected = version or cmip_version_from_env()
    if selected == "cmip7":
        path = cmip7_zarr_path or os.getenv("CMIP7_ZARR_PATH") or cmip6_zarr_path
        return CMIP7ScenarioBuilder(str(historical_zarr_path), str(path))
    return CMIP6ScenarioBuilder(str(historical_zarr_path), str(cmip6_zarr_path))


__all__ = ["CMIPVersion", "build_cmip_scenario_builder", "cmip_version_from_env"]
