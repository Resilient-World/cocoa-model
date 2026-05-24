"""Shared CMIP scenario-builder interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import xarray as xr


class CMIPScenarioBuilderBase(ABC):
    historical_zarr_path: str
    ensemble_zarr_path: str

    @abstractmethod
    def build_scenario(self, scenario: str, window: tuple[str, str]) -> xr.Dataset:
        """Return an ERA5-schema daily scenario dataset for ``scenario`` and date window."""

    @property
    @abstractmethod
    def version(self) -> str:
        """CMIP generation handled by this builder."""


__all__ = ["CMIPScenarioBuilderBase"]
