"""Pydantic models for the Avoided Loss simulation API."""

from __future__ import annotations

from enum import Enum

from typing import Literal

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


class SimulateInterventionRequest(BaseModel):
    """Request body for POST /simulate-intervention."""

    farm_location: FarmLocation
    farm_size_ha: float = Field(..., gt=0.0, description="Farm area in hectares")
    current_yield: float = Field(
        ...,
        ge=0.0,
        description="Observed current yield in tonnes per hectare",
    )
    intervention_type: InterventionType
    cocoa_price_usd: float = Field(
        ...,
        ge=0.0,
        description="Market cocoa price in USD per tonne",
    )

    # Optional cooperative-level mode: request recommendations for many farms at once.
    # When present, the API can return a per-farm ranking using CATE estimates from tabular covariates.
    batch_farms: list[dict] | None = Field(
        default=None,
        description="Optional list of farm records (cooperative rollouts); used by /rank-interventions",
    )


class RankInterventionsRequest(BaseModel):
    """Request body for POST /rank-interventions (cooperative-level targeting)."""

    rows: list[dict] = Field(..., description="Tabular rows with outcome, treatment, covariates, and farm metadata")
    outcome: str = Field(..., description="Outcome column name in rows (e.g. yield delta)")
    treatment: str = Field(..., description="Treatment indicator column name (0/1)")
    covariates: list[str] = Field(..., description="Covariate column names used for CATE")
    method: Literal["r_learner", "causal_forest"] = Field(default="r_learner")
    n_folds: int = Field(default=5, ge=2, le=10)
    cocoa_price_usd: float = Field(..., ge=0.0)
    intervention_cost_usd_per_farm: float = Field(default=0.0, ge=0.0)
    farm_area_col: str = Field(default="farm_size_ha", description="Column name for farm area in hectares")


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


class AvoidedLossInterval(BaseModel):
    """Confidence interval for avoided loss (tonnes)."""

    lower: float = Field(..., description="Lower bound (tonnes)")
    upper: float = Field(..., description="Upper bound (tonnes)")
    level: float = Field(default=0.9, description="Confidence level (e.g. 0.9 = 90%)")


class ConfidenceInterval(BaseModel):
    """Uncertainty bounds for simulation outputs."""

    avoided_loss_tonnes: AvoidedLossInterval


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
    cocoa_price_usd: float = Field(
        ...,
        ge=0.0,
        description="Market cocoa price in USD per tonne",
    )
    scenario: ScenarioSSP = Field(
        ...,
        description="CMIP6 SSP label passed through ScenarioBuilder",
    )
    horizon_year: ScenarioHorizonYear = Field(
        ...,
        description="Calendar year defining the CMIP6 climatology window for delta-change",
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
        description="USD value using mean avoided tonnes × cocoa_price_usd",
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
        description="Monetary value of avoided loss (USD)",
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

    @field_validator("avoided_loss_tonnes", "financial_impact_usd", mode="before")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        return max(0.0, float(value))


# ---------------------------------------------------------------------------
# EUDR compliance (EU) 2023/1115
# ---------------------------------------------------------------------------

from compliance.eudr import (  # noqa: E402
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
