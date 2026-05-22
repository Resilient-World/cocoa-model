"""
Mirid (Sahlbergella singularis / Distantiella theobroma) pressure proxy.

Reference: Asitoakor et al. (2024) — temperature and shade structure correlate with
mirid abundance; Asogwa et al. (2022) — ~25–30% yield attribution under high pressure.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from hazards.black_pod import ShadeSpecies

# Mirid shade modifiers (Asitoakor 2024).
SHADE_MIRID_MULTIPLIERS: dict[ShadeSpecies, float] = {
    ShadeSpecies.UNSHADED: 1.0,
    ShadeSpecies.COLA_NITIDA: 0.9,
    ShadeSpecies.KHAYA_IVORENSIS: 0.45,
    ShadeSpecies.CEDRELA_ODORATA: 0.45,
    ShadeSpecies.MILICIA_EXCELSA: 0.45,
    ShadeSpecies.TRIPLOCHITON_SCHLEROXYLON: 1.05,
}

# West Africa dry-season months for mirid peak (Nov–Mar).
_DRY_SEASON_MONTHS = {11, 12, 1, 2, 3}


class MiridPressureModel:
    """Temperature–shade proxy for mirid-driven yield loss."""

    def __init__(
        self,
        *,
        tmean_baseline_c: float = 26.0,
        tmean_scale_c: float = 4.0,
        max_attributed_loss: float = 0.25,
    ) -> None:
        self.tmean_baseline_c = tmean_baseline_c
        self.tmean_scale_c = tmean_scale_c
        self.max_attributed_loss = max_attributed_loss

    def _dry_season_tmean(self, ds: xr.Dataset) -> float:
        tmean = ds["tmean"]
        if "time" in tmean.dims or "time" in tmean.coords:
            months = tmean["time"].dt.month
            dry = tmean.where(months.isin(list(_DRY_SEASON_MONTHS)), drop=True)
            if int(dry.sizes.get("time", 0)) > 0:
                return float(dry.mean(dim="time").values)
        # Fallback: upper quartile of daily temperatures
        return float(np.quantile(np.asarray(tmean.values, dtype=np.float64).reshape(-1), 0.75))

    def annual_pressure(
        self,
        ds: xr.Dataset,
        *,
        shade_species: ShadeSpecies | str | None = None,
    ) -> float:
        t_dry = self._dry_season_tmean(ds)
        pressure = float(np.clip((t_dry - self.tmean_baseline_c) / self.tmean_scale_c, 0.0, 1.0))
        if shade_species is not None:
            species = (
                shade_species
                if isinstance(shade_species, ShadeSpecies)
                else ShadeSpecies(shade_species)
            )
            pressure *= SHADE_MIRID_MULTIPLIERS.get(species, 1.0)
        return float(np.clip(pressure, 0.0, 1.0))

    def annual_yield_loss_fraction(
        self,
        ds: xr.Dataset,
        shade_species: ShadeSpecies | str | None = None,
    ) -> float:
        pressure = self.annual_pressure(ds, shade_species=shade_species)
        return float(np.clip(self.max_attributed_loss * pressure, 0.0, 1.0))


__all__ = ["SHADE_MIRID_MULTIPLIERS", "MiridPressureModel"]
