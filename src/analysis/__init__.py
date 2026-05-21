"""Causal inference and econometric analysis utilities."""

from analysis.bjs_imputation import BJSResult, BorusyakJaravelSpiess
from analysis.csdid import ATTGTResult, ATTResult, CSEventStudyResult, CallawaySantAnna
from analysis.did_comparison_harness import compare_did_methods, write_did_comparison_report
from analysis.did_impact import (
    AvoidedRevenueResult,
    DiDResult,
    DidMethod,
    EventStudyResult,
    calculate_avoided_revenue_loss,
    calculate_did_att,
    did_estimator,
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
from analysis.parallel_trends import (
    PlaceboPretrendResult,
    goodman_bacon_decomposition,
    placebo_pretreatment_did,
)
from analysis.synthdid import SDIDResult, SyntheticDiD
from analysis.sensitivity import (
    EValueResult,
    NegativeControlResult,
    e_value,
    negative_control_outcome_test,
    rosenbaum_bounds,
    rosenbaum_gamma_at_alpha,
)

__all__ = [
    "AvoidedRevenueResult",
    "DiDResult",
    "EventStudyResult",
    "calculate_avoided_revenue_loss",
    "calculate_did_att",
    "did_estimator",
    "DidMethod",
    "compare_did_methods",
    "write_did_comparison_report",
    "SyntheticDiD",
    "SDIDResult",
    "event_study",
    "CallawaySantAnna",
    "ATTGTResult",
    "ATTResult",
    "CSEventStudyResult",
    "BorusyakJaravelSpiess",
    "BJSResult",
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
    "PlaceboPretrendResult",
    "goodman_bacon_decomposition",
    "placebo_pretreatment_did",
    "EValueResult",
    "NegativeControlResult",
    "e_value",
    "negative_control_outcome_test",
    "rosenbaum_bounds",
    "rosenbaum_gamma_at_alpha",
]
