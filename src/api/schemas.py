"""Pydantic models for the Avoided Loss simulation API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class InterventionType(str, Enum):
    """Supported farm intervention types."""

    shade_trees = "shade_trees"
    agroforestry = "agroforestry"
    drought_resistant_variety = "drought_resistant_variety"


class FarmLocation(BaseModel):
    """Geographic coordinates for a cocoa farm."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")


FinancialCurrency = Literal["USD", "GHS", "XOF", "EUR"]
PricingBasis = Literal["spot", "12m_forward", "trailing_3y_avg"]


class CurrencyFinancialBand(BaseModel):
    """Avoided-loss value in one currency with interval bounds."""

    point: float = Field(..., ge=0.0)
    ci_low: float = Field(..., ge=0.0)
    ci_high: float = Field(..., ge=0.0)
    currency: FinancialCurrency
    price_usd_per_tonne: float = Field(
        ...,
        ge=0.0,
        description="Effective cocoa price (USD/tonne) used for conversion",
    )
    pricing_basis: PricingBasis = "spot"
    farm_gate: bool = True


class FinancialImpactResponse(BaseModel):
    """Tri-currency financial impact (USD, GHS, XOF) plus primary reporting currency."""

    primary: CurrencyFinancialBand
    usd: CurrencyFinancialBand
    ghs: CurrencyFinancialBand
    xof: CurrencyFinancialBand


class SensitivityBounds(BaseModel):
    """DVDS sharp ATE partial identification bounds under the marginal sensitivity model."""

    lambda_: float = Field(..., ge=1.0, description="Odds-ratio bound Λ on unmeasured confounding")
    ate_lower: float = Field(..., description="Sharp lower bound on cooperative ATE (tonnes/ha)")
    ate_upper: float = Field(..., description="Sharp upper bound on cooperative ATE (tonnes/ha)")
    ci_lower: float = Field(..., description="95% Wald lower limit for the lower ATE bound")
    ci_upper: float = Field(..., description="95% Wald upper limit for the upper ATE bound")
    tipping_point_lambda: float | None = Field(
        default=None,
        description="Smallest Λ in [1,10] where the 95% Wald partial-ID band contains zero",
    )


MediatorId = Literal["microclimate", "soil_moisture", "cssvd_prevalence"]


class MediatorEffect(BaseModel):
    """NDE/NIE decomposition for one canonical mediator."""

    mediator: MediatorId
    nde: float
    nie: float
    total_effect: float
    proportion_mediated: float
    nde_ci: tuple[float, float]
    nie_ci: tuple[float, float]
    rho_critical: float | None = Field(
        default=None,
        description="Smallest ρ in [0,0.9] where bias-adjusted NIE ≤ 0",
    )


class MediationDecomposition(BaseModel):
    """Optional mediation block on simulate-intervention responses."""

    per_mediator: list[MediatorEffect]
    path_table: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered multi-mediator path effects when len(mediators) > 1",
    )


class SimulateInterventionRequest(BaseModel):
    """Request body for POST /simulate-intervention."""

    decompose_mediators: list[MediatorId] | None = Field(
        default=None,
        description=(
            "Optional causal mediation (NDE/NIE) for canonical paths; "
            "microclimate, soil_moisture, cssvd_prevalence. Omit for no extra cost."
        ),
    )
    include_sensitivity: bool = Field(
        default=False,
        description="Attach cooperative DVDS sensitivity bounds (requires farm panel parquet or synthetic fallback)",
    )
    farm_location: FarmLocation
    farm_size_ha: float = Field(..., gt=0.0, description="Farm area in hectares")
    current_yield: float = Field(
        ...,
        ge=0.0,
        description="Observed current yield in tonnes per hectare",
    )
    intervention_type: InterventionType
    cocoa_price_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional flat USD/tonne override; omit to use ICCO + farm-gate model",
    )
    currency: FinancialCurrency = Field(
        default="USD",
        description="Primary reporting currency for financial_impact.primary",
    )
    pricing_basis: PricingBasis = Field(
        default="spot",
        description="spot | 12m_forward (ICE curve) | trailing_3y_avg",
    )
    farm_gate: bool = Field(
        default=True,
        description="Apply country farm-gate pass-through to ICCO NY",
    )
    country_code: Literal["GHA", "CIV", "CMR"] | None = Field(
        default=None,
        description="Producer country for pass-through; inferred from coordinates if omitted",
    )
    farm_polygon: dict[str, Any] | None = Field(
        default=None,
        description="Optional GeoJSON Polygon; when set, response includes eudr_status (EUDR 2023/1115)",
    )

    # Optional cooperative-level mode: request recommendations for many farms at once.
    # When present, the API can return a per-farm ranking using CATE estimates from tabular covariates.
    batch_farms: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional list of farm records (cooperative rollouts); used by /rank-interventions",
    )


class RankInterventionsRequest(BaseModel):
    """Request body for POST /rank-interventions (cooperative-level targeting)."""

    rows: list[dict[str, Any]] = Field(
        ..., description="Tabular rows with outcome, treatment, covariates, and farm metadata"
    )
    outcome: str = Field(..., description="Outcome column name in rows (e.g. yield delta)")
    treatment: str = Field(..., description="Treatment indicator column name (0/1)")
    covariates: list[str] = Field(..., description="Covariate column names used for CATE")
    method: Literal["r_learner", "causal_forest"] = Field(default="r_learner")
    n_folds: int = Field(default=5, ge=2, le=10)
    cocoa_price_usd: float = Field(..., ge=0.0)
    intervention_cost_usd_per_farm: float = Field(default=0.0, ge=0.0)
    farm_area_col: str = Field(
        default="farm_size_ha", description="Column name for farm area in hectares"
    )


class RankedFarmRecommendation(BaseModel):
    farm_id: str | int | None = None
    net_uplift_usd: float
    gross_uplift_usd: float
    avoided_loss_tonnes: float
    tau_hat_tonnes_per_ha: float
    se: float


class RankInterventionsResponse(BaseModel):
    """Response from POST /rank-interventions."""

    method: str
    n: int
    ranked: list[RankedFarmRecommendation]


PolicyLearnerMethod = Literal["policy_tree", "policy_forest"]


class PolicyRule(BaseModel):
    """One interpretable leaf rule from a DR policy tree."""

    rule_id: int
    rule_text: str
    leaf_id: int
    n_units: int
    treat_fraction: float = Field(..., ge=0.0, le=1.0)
    expected_uplift: float
    ci_low: float
    ci_high: float


class PolicyRulebook(BaseModel):
    """Learned targeting rules with aggregate policy value."""

    method: PolicyLearnerMethod
    feature_names: list[str]
    treatment_names: list[str]
    rules: list[PolicyRule]
    policy_value_estimate: float
    policy_value_ci_low: float
    policy_value_ci_high: float
    greedy_policy_value: float | None = None
    cost_aware: bool = False
    n_samples: int


class LearnPolicyRulesRequest(BaseModel):
    """Request body for POST /learn-policy-rules."""

    rows: list[dict[str, Any]] = Field(
        ..., description="Panel rows with outcome, treatment, and covariates"
    )
    outcome: str
    treatment: str
    covariates: list[str]
    learner: Literal["tree", "forest"] = Field(
        default="tree",
        description="policy_tree (fast) or policy_forest (500 trees, offline/HPC)",
    )
    cost_col: str | None = Field(
        default=None,
        description="Per-unit cost column; enables net-benefit (uplift − cost) policy learning",
    )
    farm_area_col: str = Field(default="farm_size_ha")
    cocoa_price_usd: float = Field(default=3000.0, ge=0.0)
    intervention_cost_usd_per_farm: float = Field(default=0.0, ge=0.0)
    budget: float | None = Field(
        default=None,
        ge=0.0,
        description="Budget for greedy CATE/cost baseline comparison",
    )
    max_depth: int = Field(default=4, ge=1, le=12)
    min_samples_leaf: int = Field(default=50, ge=10)
    n_estimators: int = Field(default=500, ge=10, le=2000)
    n_folds: int = Field(default=5, ge=2, le=10)
    n_bootstrap: int = Field(default=100, ge=20, le=500)
    random_state: int = 42
    recommended_treatment_label: str = Field(
        default="treat",
        description="Label embedded in rendered rules (e.g. treat_with_shade_trees)",
    )
    cate_method: Literal["r_learner", "causal_forest"] = Field(default="r_learner")


class LearnPolicyRulesResponse(BaseModel):
    """Response from POST /learn-policy-rules."""

    rulebook: PolicyRulebook


UQMethod = Literal["mcd", "cqr"]

ConfidenceMethod = Literal[
    "mcd",
    "cqr",
    "split_cqr",
    "aci",
    "conformal_pid",
    "eci",
    "eci_integral",
]


class AvoidedLossInterval(BaseModel):
    """Confidence interval for avoided loss (tonnes)."""

    lower: float = Field(..., description="Lower bound (tonnes)")
    upper: float = Field(..., description="Upper bound (tonnes)")
    level: float = Field(default=0.9, description="Confidence level (e.g. 0.9 = 90%)")


DriftDiagnosis = Literal["none", "covariate_shift", "concept_shift", "out_of_support"]


class DriftStatus(BaseModel):
    """Dashboard payload for per-stratum drift monitoring."""

    stratum_key: str
    log_martingale: float
    alarm_active: bool
    diagnosis: DriftDiagnosis
    coverage_running_avg: float | None = None


class DriftAlarmPayload(BaseModel):
    """Active drift alarm attached to /simulate-scenario when detected."""

    type: Literal["covariate_shift", "concept_shift", "out_of_support"]
    log_martingale: float
    triggered_at: str


class ConfidenceInterval(BaseModel):
    """Uncertainty bounds for simulation outputs."""

    avoided_loss_tonnes: AvoidedLossInterval
    method: ConfidenceMethod = Field(
        default="mcd",
        description="Uncertainty method: MC dropout, CQR, or online conformal (scenario API)",
    )
    empirical_coverage: float | None = Field(
        default=None,
        description="Empirical coverage on calibration split when method=cqr (if known)",
    )
    coverage_running_avg: float | None = Field(
        default=None,
        description="Rolling mean coverage over the last 1000 online updates for this stratum",
    )
    cv_strategy: str | None = Field(
        default=None,
        description="CQR calibration CV strategy (e.g. spatial_block); omitted for legacy calibrators",
    )


class ConformalIntervalResponse(BaseModel):
    """Split / Mondrian conformal interval with finite-sample coverage statement."""

    point: float = Field(..., description="Point prediction (tonnes/ha or tonnes)")
    lower: float = Field(..., description="Lower conformal bound")
    upper: float = Field(..., description="Upper conformal bound")
    coverage_target: float = Field(
        default=0.9,
        description="Target marginal coverage (1 − α)",
    )
    method: str = Field(
        default="split_conformal",
        description="Conformal method identifier",
    )
    coverage_guarantee: str = Field(
        ...,
        description="Human-readable finite-sample coverage guarantee",
    )


class ConformalConfidenceInterval(BaseModel):
    """Conformal prediction intervals (90% by default when α=0.1)."""

    baseline_yield_tonnes_per_ha: ConformalIntervalResponse | None = None
    projected_yield_tonnes_per_ha: ConformalIntervalResponse | None = None
    avoided_loss_tonnes: ConformalIntervalResponse | None = None


ScenarioSSP = Literal["ssp245", "ssp585"]
ScenarioHorizonYear = Literal[2030, 2050, 2080]
DownscalingMethod = Literal["linear_delta", "corrdiff", "neuralgcm", "ace2_era5", "aurora"]

ProcessEnsembleMethod = Literal["mean", "bma", "best"]


class SimulateScenarioRequest(BaseModel):
    """Request body for POST /simulate-scenario (SSP × horizon climate + intervention)."""

    farm_location: FarmLocation
    farm_size_ha: float = Field(..., gt=0.0, description="Farm area in hectares")
    current_yield: float = Field(
        ...,
        ge=0.0,
        description="Observed current yield in tonnes per hectare",
    )
    intervention_type: InterventionType
    cocoa_price_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional flat USD/tonne override; omit to use ICCO + farm-gate model",
    )
    currency: FinancialCurrency = Field(default="USD")
    pricing_basis: PricingBasis = Field(default="spot")
    farm_gate: bool = Field(default=True)
    country_code: Literal["GHA", "CIV", "CMR"] | None = None
    farm_polygon: dict[str, Any] | None = Field(
        default=None,
        description="Optional GeoJSON Polygon; when set, response includes eudr_status",
    )
    scenario: ScenarioSSP = Field(
        ...,
        description="CMIP6 SSP label passed through ScenarioBuilder",
    )
    horizon_year: ScenarioHorizonYear = Field(
        ...,
        description="Calendar year defining the CMIP6 climatology window for delta-change",
    )
    downscaling_method: DownscalingMethod = Field(
        default="linear_delta",
        description=(
            "linear_delta, corrdiff, neuralgcm, ace2_era5, or aurora "
            "(latter three require env flags; aurora also needs AURORA_COMMERCIAL_OK in production)"
        ),
    )
    ensemble_process_method: ProcessEnsembleMethod = Field(
        default="mean",
        description="Combine CASEJ/CASE2/ALMANAC when PROCESS_BMA_ENABLED=true",
    )


class YieldUncertaintyBand(BaseModel):
    """Monte Carlo summaries from paired forwards through YieldSurrogateModel."""

    mean: float = Field(..., description="Mean yield (blended with observed yield when configured)")
    p10: float = Field(..., description="10th percentile of MC yields (tonnes/ha)")
    p90: float = Field(..., description="90th percentile of MC yields (tonnes/ha)")


class AvoidedLossUncertaintyBand(BaseModel):
    """Distribution of avoided loss (tonnes) from MC yield pairs."""

    mean: float = Field(..., ge=0.0)
    p10: float = Field(..., ge=0.0)
    p90: float = Field(..., ge=0.0)


class SimulateClimateAttributionRequest(BaseModel):
    """Request body for POST /simulate-climate-attribution (factual vs ATTRICI counterfactual)."""

    farm_location: FarmLocation
    farm_size_ha: float = Field(..., gt=0.0)
    current_yield: float = Field(..., ge=0.0)
    intervention_type: InterventionType
    cocoa_price_usd: float | None = Field(default=None, ge=0.0)
    currency: FinancialCurrency = Field(default="USD")
    pricing_basis: PricingBasis = Field(default="spot")
    farm_gate: bool = Field(default=True)
    country_code: Literal["GHA", "CIV", "CMR"] | None = None
    climate_year: int | None = Field(
        default=None,
        description="Calendar year for ERA5 / counterfactual slice (default: API CLIMATE_REFERENCE_YEAR)",
    )


class SimulateClimateAttributionResponse(BaseModel):
    """
    Decomposes avoided loss into climate-attributed vs intervention components.

    ``attributed_loss_tonnes_per_ha = counterfactual_yield - factual_yield`` (climate-change
    impact on yield). ``total_avoided_loss_tonnes`` combines climate buffer + intervention uplift.
    """

    factual_yield_tonnes_per_ha: float
    counterfactual_yield_tonnes_per_ha: float
    attributed_loss_tonnes_per_ha: float = Field(
        ...,
        description="Per-ha yield gap: no-climate-change world minus current (≥ 0 when warming hurt)",
    )
    intervention_avoided_loss_tonnes: float = Field(
        ...,
        ge=0.0,
        description="Farm-level avoided loss from intervention (existing simulate-intervention logic)",
    )
    total_avoided_loss_tonnes: float = Field(
        ...,
        ge=0.0,
        description="attributed_loss × area + intervention_avoided_loss",
    )
    climate_reference_year: int
    financial_impact_usd: float = Field(..., ge=0.0)
    financial_impact: FinancialImpactResponse


class ScenarioSourceAttribution(BaseModel):
    """Provenance entry for scenario simulation (models, data, regulations)."""

    id: str
    role: str
    citation: str | None = None
    asset: str | None = None
    aurora_model_version: str | None = Field(
        default=None,
        description="Aurora checkpoint id when downscaling_method=aurora",
    )
    aurora_lora_id: str | None = Field(
        default=None,
        description="Per-region LoRA adapter id (base or region key)",
    )


class SimulateScenarioResponse(BaseModel):
    """Response from POST /simulate-scenario."""

    scenario: ScenarioSSP
    horizon_year: ScenarioHorizonYear
    climate_reference_year: int = Field(
        ...,
        description="Calendar year slice from the adjusted ERA5 daily stack",
    )
    baseline_yield_tonnes_per_ha: YieldUncertaintyBand = Field(
        ...,
        description="Yield under SSP-conditioned climate without intervention encoding",
    )
    projected_yield_tonnes_per_ha: YieldUncertaintyBand = Field(
        ...,
        description="Yield under SSP-conditioned climate with intervention",
    )
    avoided_loss_tonnes: AvoidedLossUncertaintyBand
    financial_impact_usd_mean: float = Field(
        ...,
        ge=0.0,
        description="USD point estimate (equals financial_impact.usd.point)",
    )
    financial_impact: FinancialImpactResponse = Field(
        ...,
        description="Avoided-loss valuation in USD, GHS, and XOF",
    )
    confidence_interval: ConfidenceInterval | None = Field(
        default=None,
        description="Avoided-loss conformal interval (online ECI-Integral by default)",
    )
    drift_alarm: DriftAlarmPayload | None = Field(
        default=None,
        description="WCTM drift alarm when non-stationarity is detected",
    )
    drift_status: DriftStatus | None = Field(
        default=None,
        description="Current WCTM state for this scenario stratum (dashboard)",
    )
    eudr_status: EudrStatusBlock | None = Field(
        default=None,
        description="Present when request includes farm_polygon (EUDR Art. 3 / Whisp)",
    )
    downscaling_method: DownscalingMethod = Field(
        default="linear_delta",
        description="Climate downscaling path used for this response",
    )
    corrdiff_samples_used: int | None = Field(
        default=None,
        description="Number of CorrDiff ensemble members when downscaling_method=corrdiff",
    )
    source_attributions: list[ScenarioSourceAttribution] = Field(
        default_factory=list,
        description="Model and data provenance (includes Aurora fields when applicable)",
    )


class BioticLossAttribution(BaseModel):
    """Per-pathogen yield-loss fractions before multiplicative survival."""

    black_pod: float = Field(..., ge=0.0, le=1.0)
    cssvd: float = Field(..., ge=0.0, le=1.0)
    mirids: float = Field(..., ge=0.0, le=1.0)


class ScenarioBioticLosses(BaseModel):
    """Biotic survival and attribution for one climate/static path."""

    surviving_fraction: float = Field(..., ge=0.0, le=1.0)
    total_loss_fraction: float = Field(..., ge=0.0, le=1.0)
    loss_attribution: BioticLossAttribution


class SimulateInterventionResponse(BaseModel):
    """Response from POST /simulate-intervention."""

    baseline_yield_tonnes_per_ha: float = Field(
        ...,
        description="Counterfactual yield without intervention (tonnes/ha)",
    )
    projected_yield_tonnes_per_ha: float = Field(
        ...,
        description="Factual yield with intervention (tonnes/ha)",
    )
    avoided_loss_tonnes: float = Field(
        ...,
        ge=0.0,
        description="Total avoided yield loss for the farm (tonnes)",
    )
    financial_impact_usd: float = Field(
        ...,
        ge=0.0,
        description="USD point estimate (equals financial_impact.usd.point)",
    )
    financial_impact: FinancialImpactResponse = Field(
        ...,
        description="Avoided-loss valuation in USD, GHS, and XOF",
    )
    confidence_interval: ConfidenceInterval
    conformal_interval: ConformalConfidenceInterval | None = Field(
        default=None,
        description="Present when models/conformal.json is loaded at API startup",
    )
    biotic_loss_attribution: dict[str, ScenarioBioticLosses] | None = Field(
        default=None,
        description=(
            "Counterfactual (baseline) and projected biotic loss fractions; "
            "keys ``baseline`` and ``projected``"
        ),
    )
    eudr_status: EudrStatusBlock | None = Field(
        default=None,
        description="Present when request includes farm_polygon (EUDR Art. 3 / Whisp)",
    )
    sensitivity_bounds: list[SensitivityBounds] | None = Field(
        default=None,
        description=(
            "Cooperative observational ATE bounds under Tan's MSM (DVDS); "
            "not the per-farm MC/CQR interval. Present when include_sensitivity=true."
        ),
    )
    mediation: MediationDecomposition | None = Field(
        default=None,
        description="Present when request.decompose_mediators is non-empty",
    )

    @field_validator("avoided_loss_tonnes", "financial_impact_usd", mode="before")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        return max(0.0, float(value))


# ---------------------------------------------------------------------------
# EUDR compliance (EU) 2023/1115
# ---------------------------------------------------------------------------

from api.eudr import EudrStatusBlock
from compliance.eudr import (
    DueDiligenceStatement,
    OperatorInfo,
    PlotGeometry,
    ProductInfo,
    RiskScore,
)


class ComplianceDdsRequest(BaseModel):
    """Request body for POST /compliance/dds."""

    plot: PlotGeometry
    operator: OperatorInfo
    product: ProductInfo
    buyer_name: str | None = None
    supplier_name: str | None = None
    supply_chain_complexity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Normalised supply-chain complexity (Art. 10)",
    )
    use_gee_deforestation_check: bool = Field(
        default=False,
        description="When false, skips live GEE screening (geolocation + risk only)",
    )


class ComplianceDdsResponse(BaseModel):
    """Due diligence statement and risk score for a cocoa plot."""

    dds: DueDiligenceStatement
    dds_json: str = Field(..., description="Serialised DDS (JSON)")
    dds_csv: str = Field(..., description="EU Information System CSV row(s)")
    risk_score: RiskScore
    geolocation_valid: bool
    validation_errors: list[str] = Field(default_factory=list)
