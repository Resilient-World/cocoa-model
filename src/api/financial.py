"""Financial valuation helpers for single-farm simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

if TYPE_CHECKING:
    from api.schemas import FinancialImpactResponse
    from models.cocoa_quality import CocoaQualityPrediction

from finance.pricing import (
    PricingBasis,
    convert_usd_amount,
    infer_country_code,
    resolve_price_usd_per_tonne,
)

CurrencyCode = Literal["USD", "GHS", "XOF", "EUR"]
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FinancialImpact:
    """Monetary value of avoided loss in a single currency."""

    point: float
    ci_low: float
    ci_high: float
    currency: CurrencyCode
    price_usd_per_tonne: float
    pricing_basis: PricingBasis
    farm_gate: bool


@dataclass(frozen=True)
class FinancialImpactMulti:
    """Avoided-loss valuation in USD, GHS, and XOF (underwriter tri-currency view)."""

    primary: FinancialImpact
    usd: FinancialImpact
    ghs: FinancialImpact
    xof: FinancialImpact


def calculate_financial_impact_usd(
    avoided_loss_tonnes: float,
    cocoa_price_usd: float,
) -> float:
    """Legacy: convert avoided tonnes × flat USD/tonne price."""
    return max(0.0, avoided_loss_tonnes) * max(0.0, cocoa_price_usd)


def calculate_financial_impact(
    avoided_loss_tonnes: float,
    *,
    currency: CurrencyCode = "USD",
    pricing_basis: PricingBasis = "spot",
    farm_gate: bool = True,
    country_code: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    cocoa_price_usd: float | None = None,
    ci_low_tonnes: float | None = None,
    ci_high_tonnes: float | None = None,
    quality_premium_usd_per_t: float = 0.0,
) -> FinancialImpactMulti:
    """
    Value avoided loss using ICCO / forward / trailing avg and FX conversion.

    Parameters
    ----------
    avoided_loss_tonnes:
        Point estimate of avoided yield (tonnes).
    currency:
        Primary reporting currency for :attr:`FinancialImpactMulti.primary`.
    pricing_basis:
        ``spot``, ``12m_forward`` (ICE-style 12m leg), or ``trailing_3y_avg``.
    farm_gate:
        Apply country pass-through to ICCO NY when True.
    country_code:
        ISO3 producer code (``GHA``, ``CIV``, ``CMR``); inferred from lat/lon if omitted.
    cocoa_price_usd:
        Optional override of market USD/tonne (backward compatible with flat pricing).
    ci_low_tonnes, ci_high_tonnes:
        Optional avoided-loss interval for financial CI (defaults to point).
    """
    if country_code is None:
        if lat is not None and lon is not None:
            country_code = infer_country_code(lat, lon)
        else:
            country_code = "CIV"

    price_usd = resolve_price_usd_per_tonne(
        pricing_basis=pricing_basis,
        farm_gate=farm_gate,
        country_code=country_code,
        price_override_usd=cocoa_price_usd,
    )
    price_usd = max(0.0, price_usd + quality_premium_usd_per_t)

    tonnes_low = ci_low_tonnes if ci_low_tonnes is not None else avoided_loss_tonnes
    tonnes_high = ci_high_tonnes if ci_high_tonnes is not None else avoided_loss_tonnes

    usd_point = max(0.0, avoided_loss_tonnes) * price_usd
    usd_low = max(0.0, tonnes_low) * price_usd
    usd_high = max(0.0, tonnes_high) * price_usd

    def _pack(
        amount_usd: float, low_usd: float, high_usd: float, code: CurrencyCode
    ) -> FinancialImpact:
        return FinancialImpact(
            point=convert_usd_amount(amount_usd, code),
            ci_low=convert_usd_amount(low_usd, code),
            ci_high=convert_usd_amount(high_usd, code),
            currency=code,
            price_usd_per_tonne=price_usd,
            pricing_basis=pricing_basis,
            farm_gate=farm_gate,
        )

    usd = _pack(usd_point, usd_low, usd_high, "USD")
    ghs = _pack(usd_point, usd_low, usd_high, "GHS")
    xof = _pack(usd_point, usd_low, usd_high, "XOF")
    primary = _pack(usd_point, usd_low, usd_high, currency)

    return FinancialImpactMulti(primary=primary, usd=usd, ghs=ghs, xof=xof)


def load_quality_premium_map(path: Path | None = None) -> dict[str, float]:
    premium_path = path or _REPO_ROOT / "config" / "quality_premiums.yaml"
    if not premium_path.is_file():
        return {
            "fine_flavor_probability_threshold": 0.7,
            "fine_flavor_usd_per_t": 1200.0,
            "controlled_fermentation_usd_per_t": 150.0,
            "defect_rate_penalty_threshold_pct": 5.0,
            "defect_rate_penalty_usd_per_t": -400.0,
        }
    with premium_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(key): float(value) for key, value in data.items()}


def quality_price_premium_usd_per_t(
    quality: CocoaQualityPrediction,
    *,
    premium_map: dict[str, float] | None = None,
) -> float:
    premiums = premium_map or load_quality_premium_map()
    premium = 0.0
    if quality.fine_flavor_probability >= premiums.get("fine_flavor_probability_threshold", 0.7):
        premium += premiums.get("fine_flavor_usd_per_t", 1200.0)
    if quality.fermentation_index >= 0.65:
        premium += premiums.get("controlled_fermentation_usd_per_t", 150.0)
    if quality.defect_rate > premiums.get("defect_rate_penalty_threshold_pct", 5.0):
        premium += premiums.get("defect_rate_penalty_usd_per_t", -400.0)
    return float(premium)


def financial_impact_to_schema(block: FinancialImpactMulti) -> FinancialImpactResponse:
    """Map dataclass bundle to Pydantic response (avoids circular import at module load)."""
    from api.schemas import CurrencyFinancialBand, FinancialImpactResponse

    def _one(impact: FinancialImpact) -> CurrencyFinancialBand:
        return CurrencyFinancialBand(
            point=impact.point,
            ci_low=impact.ci_low,
            ci_high=impact.ci_high,
            currency=impact.currency,
            price_usd_per_tonne=impact.price_usd_per_tonne,
            pricing_basis=impact.pricing_basis,
            farm_gate=impact.farm_gate,
        )

    return FinancialImpactResponse(
        primary=_one(block.primary),
        usd=_one(block.usd),
        ghs=_one(block.ghs),
        xof=_one(block.xof),
    )
