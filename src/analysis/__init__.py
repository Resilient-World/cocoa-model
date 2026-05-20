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
    AIPWResult,
    BalanceReport,
    aipw_estimator,
    compute_propensity_scores,
    default_logit_caliper,
    love_plot_data,
    match_nearest_neighbor,
    propensity_score_match,
    standardized_mean_differences,
    trim_common_support,
)
from analysis.sensitivity import EValueResult, e_value, rosenbaum_bounds

__all__ = [
    "AvoidedRevenueResult",
    "DiDResult",
    "EventStudyResult",
    "calculate_avoided_revenue_loss",
    "calculate_did_att",
    "event_study",
    "AIPWResult",
    "BalanceReport",
    "aipw_estimator",
    "compute_propensity_scores",
    "default_logit_caliper",
    "love_plot_data",
    "match_nearest_neighbor",
    "propensity_score_match",
    "standardized_mean_differences",
    "trim_common_support",
    "EValueResult",
    "e_value",
    "rosenbaum_bounds",
]
