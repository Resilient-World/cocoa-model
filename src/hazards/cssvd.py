"""
Cocoa swollen shoot virus disease (CSSVD) yield-loss model.

CSSVD prevalence is treated as a slow exogenous process (CRIG survey rasters in
production; mock/static covariate for now). Optional
:class:`~hazards.cssvd_landscape.LandscapeCSSVDModel` maps landscape drivers
(Dumont et al. 2025) to 12-month incidence before Ofori yield loss.

Reference: Ofori et al. (2022) — ~21.17% dry-bean yield (DBY) reduction at scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazards.cssvd_landscape import IncidencePrediction, LandscapeCSSVDModel


class CSSVDRiskModel:
    """
    Map farm-level CSSVD prevalence (%) to annual yield-loss fraction.

    Tolerant clones (e.g. T63/971, T17/358) use ``tolerance`` < 1.0 on the loss term.
    """

    def __init__(
        self,
        *,
        dby_reduction_at_full_prevalence: float = 0.2117,
        landscape_model: LandscapeCSSVDModel | None = None,
    ) -> None:
        self.dby_reduction_at_full_prevalence = dby_reduction_at_full_prevalence
        self.landscape_model = landscape_model

    def annual_yield_loss_fraction(
        self,
        prevalence_pct: float,
        *,
        tolerance: float = 1.0,
    ) -> float:
        """
        Parameters
        ----------
        prevalence_pct:
            Farm-level CSSVD prevalence in percent (0–100).
        tolerance:
            Clone genetics modifier (1.0 = susceptible; ~0.3 for tolerant clones).
        """
        prev = max(0.0, min(100.0, float(prevalence_pct)))
        tol = max(0.0, float(tolerance))
        return float(self.dby_reduction_at_full_prevalence * (prev / 100.0) * tol)

    def annual_yield_loss_from_landscape(
        self,
        lat: float,
        lon: float,
        year: int,
        *,
        tolerance: float = 1.0,
        conservative: bool = False,
        landscape_model: LandscapeCSSVDModel | None = None,
    ) -> tuple[float, IncidencePrediction | None]:
        """
        Predict 12-month incidence from landscape covariates, then map to yield loss.

        When ``conservative=True``, uses the upper bound of the 90% incidence PI.
        """
        from hazards.cssvd_landscape import IncidencePrediction

        model = landscape_model or self.landscape_model
        if model is None:
            raise RuntimeError("No LandscapeCSSVDModel configured")

        inc: IncidencePrediction = model.predict_12mo_incidence(lat, lon, year)
        prob = inc.pi_high if conservative else inc.point
        prevalence_pct = 100.0 * prob
        loss = self.annual_yield_loss_fraction(prevalence_pct, tolerance=tolerance)
        return loss, inc

    @classmethod
    def with_landscape_checkpoint(
        cls,
        checkpoint: Path | str,
        *,
        dby_reduction_at_full_prevalence: float = 0.2117,
    ) -> CSSVDRiskModel:
        """Construct model with :class:`~hazards.cssvd_landscape.LandscapeCSSVDModel` loaded."""
        from hazards.cssvd_landscape import LandscapeCSSVDModel

        return cls(
            dby_reduction_at_full_prevalence=dby_reduction_at_full_prevalence,
            landscape_model=LandscapeCSSVDModel.from_checkpoint(checkpoint),
        )


__all__ = ["CSSVDRiskModel"]
