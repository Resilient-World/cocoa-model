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

    @field_validator("avoided_loss_tonnes", "financial_impact_usd", mode="before")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        return max(0.0, float(value))
