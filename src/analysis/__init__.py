"""Causal inference and econometric analysis utilities."""

from analysis.did_impact import (
    AvoidedRevenueResult,
    DiDResult,
    EventStudyResult,
    calculate_avoided_revenue_loss,
    calculate_did_att,
    event_study,
)
from analysis.psm_matching import (
    compute_propensity_scores,
    match_nearest_neighbor,
    propensity_score_match,
)

__all__ = [
    "AvoidedRevenueResult",
    "DiDResult",
    "EventStudyResult",
    "calculate_avoided_revenue_loss",
    "calculate_did_att",
    "event_study",
    "compute_propensity_scores",
    "match_nearest_neighbor",
    "propensity_score_match",
]
