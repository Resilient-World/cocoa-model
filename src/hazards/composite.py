"""
Composite biotic yield-loss layer (black pod + CSSVD + mirids).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from hazards.black_pod import BlackPodRiskModel, ShadeSpecies
from hazards.cssvd import CSSVDRiskModel
from hazards.mirids import MiridPressureModel

MIN_SURVIVING_FRACTION = 0.30  # cap combined loss at 70%


def _resolve_shade_species(static_features: dict[str, Any]) -> ShadeSpecies | None:
    raw = static_features.get("shade_species")
    if raw is None:
        return ShadeSpecies.UNSHADED
    if isinstance(raw, ShadeSpecies):
        return raw
    return ShadeSpecies(str(raw))


def apply_biotic_losses(
    climate_yield_t_ha: float,
    ds: xr.Dataset,
    static_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply independent multiplicative biotic survival factors to climate-only yield.

    Parameters
    ----------
    climate_yield_t_ha:
        Yield from the climate surrogate (tonnes/ha) before biotic adjustment.
    ds:
        Daily climate ``xr.Dataset`` with ``tmean``, ``rh_mean``, ``precip``.
    static_features:
        Farm static covariates, e.g. ``cssvd_prevalence_pct``, ``cssvd_tolerance``,
        ``shade_species``.

    Returns
    -------
    dict with ``final_yield``, ``surviving_fraction``, and ``loss_attribution`` fractions.
    """
    shade = _resolve_shade_species(static_features)

    bp_model = BlackPodRiskModel()
    bp_loss = float(bp_model.seasonal_yield_loss_fraction(ds, shade_species=shade).values)

    cssvd_model = CSSVDRiskModel()
    prevalence = float(static_features.get("cssvd_prevalence_pct", 15.0))
    tolerance = float(static_features.get("cssvd_tolerance", 1.0))
    cssvd_loss = cssvd_model.annual_yield_loss_fraction(prevalence, tolerance=tolerance)

    mirid_model = MiridPressureModel()
    mirid_loss = mirid_model.annual_yield_loss_fraction(ds, shade_species=shade)

    surviving_fraction = (1.0 - bp_loss) * (1.0 - cssvd_loss) * (1.0 - mirid_loss)
    surviving_fraction = max(float(surviving_fraction), MIN_SURVIVING_FRACTION)

    final_yield = float(climate_yield_t_ha) * surviving_fraction
    total_loss_fraction = 1.0 - surviving_fraction

    return {
        "final_yield": final_yield,
        "climate_yield_t_ha": float(climate_yield_t_ha),
        "surviving_fraction": surviving_fraction,
        "total_loss_fraction": total_loss_fraction,
        "loss_attribution": {
            "black_pod": bp_loss,
            "cssvd": cssvd_loss,
            "mirids": mirid_loss,
        },
    }


def estimate_surviving_biotic_fraction(
    ds: xr.Dataset,
    static_features: dict[str, Any] | None = None,
) -> float:
    """Multiplicative biotic survival used to back out pre-biotic yield targets."""
    features = static_features or {}
    return float(apply_biotic_losses(1.0, ds, features)["surviving_fraction"])


__all__ = [
    "apply_biotic_losses",
    "estimate_surviving_biotic_fraction",
    "MIN_SURVIVING_FRACTION",
]
