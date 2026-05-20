"""Pydantic models for the Avoided Loss simulation API."""

from __future__ import annotations

from enum import Enum

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
