"""
Climate-driven black pod (Phytophthora megakarya / P. palmivora) risk model.

References
----------
- Etaware et al. (2020): RH ≥ 75% favors pathogen establishment.
- Ndoumbe-Nkeng et al. (2009): rainfall correlation with pod rot incidence.
- Asogwa et al. (2022): reported 30–90% yield loss range under severe epidemics.
- Asitoakor et al. (2024): shade species modify microclimate and disease pressure.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import xarray as xr


class ShadeSpecies(str, Enum):
    """Overstory / shade-tree species affecting black-pod microclimate."""

    UNSHADED = "unshaded"
    COLA_NITIDA = "cola_nitida"
    KHAYA_IVORENSIS = "khaya_ivorensis"
    CEDRELA_ODORATA = "cedrela_odorata"
    MILICIA_EXCELSA = "milicia_excelsa"
    TRIPLOCHITON_SCHLEROXYLON = "triplochiton_scleroxylon"


# Black-pod pressure multipliers (Asitoakor 2024): Cola nitida worse, Khaya protective.
SHADE_BLACK_POD_MULTIPLIERS: dict[ShadeSpecies, float] = {
    ShadeSpecies.UNSHADED: 1.0,
    ShadeSpecies.COLA_NITIDA: 1.4,
    ShadeSpecies.KHAYA_IVORENSIS: 0.7,
    ShadeSpecies.CEDRELA_ODORATA: 0.75,
    ShadeSpecies.MILICIA_EXCELSA: 0.75,
    ShadeSpecies.TRIPLOCHITON_SCHLEROXYLON: 1.05,
}


def _sigmoid(x: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
    if isinstance(x, xr.DataArray):
        return 1.0 / (1.0 + np.exp(-x))
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class BlackPodRiskModel:
    """
    Daily humidity–temperature–rainfall infection index with seasonal aggregation.

    Default loss curve: ``0.30 + 0.60 * sigmoid((pressure - threshold) / scale)``,
    capped at 90% yield loss (Asogwa 2022 range).
    """

    def __init__(
        self,
        *,
        rh_threshold_pct: float = 75.0,
        tmean_optimal_c: float = 22.0,
        precip_wet_mm: float = 5.0,
        rolling_days: int = 30,
        pressure_threshold: float = 8.0,
        pressure_scale: float = 2.0,
        min_loss: float = 0.30,
        loss_span: float = 0.60,
        max_loss: float = 0.90,
    ) -> None:
        self.rh_threshold_pct = rh_threshold_pct
        self.tmean_optimal_c = tmean_optimal_c
        self.precip_wet_mm = precip_wet_mm
        self.rolling_days = rolling_days
        self.pressure_threshold = pressure_threshold
        self.pressure_scale = pressure_scale
        self.min_loss = min_loss
        self.loss_span = loss_span
        self.max_loss = max_loss

    def daily_infection_index(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Per-day infection propensity in [0, 1].

        ``sigmoid((RH - 75) / 5) * sigmoid((tmean - 22) / 2) * (precip > 5 mm)``.
        """
        if "rh_mean" not in ds:
            raise KeyError("Dataset needs rh_mean for black pod risk")
        rh = ds["rh_mean"]
        tmean = ds["tmean"]
        precip = ds["precip"]

        rh_term = _sigmoid((rh - self.rh_threshold_pct) / 5.0)
        temp_term = _sigmoid((tmean - self.tmean_optimal_c) / 2.0)
        wet = (precip > self.precip_wet_mm).astype(np.float32)
        daily = (rh_term * temp_term * wet).astype(np.float32)
        return daily.rename("black_pod_daily_index")

    def seasonal_pressure(
        self,
        ds: xr.Dataset,
        *,
        shade_species: ShadeSpecies | str | None = None,
    ) -> float:
        """Peak 30-day rolling sum of daily index, optionally shade-adjusted."""
        daily = self.daily_infection_index(ds)
        roll = daily.rolling(time=self.rolling_days, min_periods=1).sum()
        pressure = float(roll.max(dim="time").values)
        if shade_species is not None:
            species = (
                shade_species
                if isinstance(shade_species, ShadeSpecies)
                else ShadeSpecies(shade_species)
            )
            pressure *= SHADE_BLACK_POD_MULTIPLIERS.get(species, 1.0)
        return pressure

    def seasonal_yield_loss_fraction(
        self,
        ds: xr.Dataset,
        shade_species: ShadeSpecies | str | None = None,
    ) -> xr.DataArray:
        """Scalar seasonal yield-loss fraction in [min_loss, max_loss]."""
        pressure = self.seasonal_pressure(ds, shade_species=shade_species)
        raw = self.min_loss + self.loss_span * float(
            _sigmoid((pressure - self.pressure_threshold) / self.pressure_scale)
        )
        loss = float(np.clip(raw, 0.0, self.max_loss))
        return xr.DataArray(loss, name="black_pod_yield_loss_fraction")


__all__ = [
    "BlackPodRiskModel",
    "ShadeSpecies",
    "SHADE_BLACK_POD_MULTIPLIERS",
]
