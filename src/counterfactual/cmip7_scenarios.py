"""CMIP7 placeholder builder for AR7-era scenario readiness."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import xarray as xr

from counterfactual.cmip_scenarios_base import CMIPScenarioBuilderBase

CMIP7_SCENARIOS = ("SSP1-1.9", "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5")
CMIP7_HORIZONS = (2030, 2050, 2080, 2100)

log = logging.getLogger(__name__)


@dataclass
class CMIP7ScenarioBuilder(CMIPScenarioBuilderBase):
    historical_zarr_path: str
    cmip7_zarr_path: str

    def __post_init__(self) -> None:
        self.ensemble_zarr_path = self.cmip7_zarr_path

    @property
    def version(self) -> str:
        return "cmip7"

    def build_scenario(self, scenario: str, window: tuple[str, str]) -> xr.Dataset:
        if scenario not in CMIP7_SCENARIOS:
            raise ValueError(
                f"Unsupported CMIP7 scenario '{scenario}'. Expected one of {CMIP7_SCENARIOS}."
            )
        msg = (
            f"CMIP7 ensemble not yet published on the configured path: {self.cmip7_zarr_path}. "
            "Set CMIP_VERSION=cmip6 until the AR7 ensemble Zarr is available."
        )
        log.warning(msg)
        raise FileNotFoundError(msg)


__all__ = ["CMIP7_HORIZONS", "CMIP7_SCENARIOS", "CMIP7ScenarioBuilder"]
