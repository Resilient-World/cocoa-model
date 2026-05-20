"""
Cocoa swollen shoot virus disease (CSSVD) yield-loss model.

CSSVD prevalence is treated as a slow exogenous process (CRIG survey rasters in
production; mock/static covariate for now).

Reference: Ofori et al. (2022) — ~21.17% dry-bean yield (DBY) reduction at scale.
"""

from __future__ import annotations


class CSSVDRiskModel:
    """
    Map farm-level CSSVD prevalence (%) to annual yield-loss fraction.

    Tolerant clones (e.g. T63/971, T17/358) use ``tolerance`` < 1.0 on the loss term.
    """

    def __init__(self, *, dby_reduction_at_full_prevalence: float = 0.2117) -> None:
        self.dby_reduction_at_full_prevalence = dby_reduction_at_full_prevalence

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


__all__ = ["CSSVDRiskModel"]
