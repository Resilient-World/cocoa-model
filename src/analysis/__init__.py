"""Causal inference and econometric analysis utilities."""

from analysis.did_impact import (
    AvoidedRevenueResult,
    DiDResult,
    calculate_avoided_revenue_loss,
    calculate_did_att,
)
from analysis.psm_matching import (
    compute_propensity_scores,
    match_nearest_neighbor,
    propensity_score_match,
)

__all__ = [
    "AvoidedRevenueResult",
    "DiDResult",
    "calculate_avoided_revenue_loss",
    "calculate_did_att",
    "compute_propensity_scores",
    "match_nearest_neighbor",
    "propensity_score_match",
]
