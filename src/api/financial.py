"""Financial valuation helpers for single-farm simulation."""

from __future__ import annotations


def calculate_financial_impact_usd(
    avoided_loss_tonnes: float,
    cocoa_price_usd: float,
) -> float:
    """Convert total avoided yield loss (tonnes) to USD."""
    return max(0.0, avoided_loss_tonnes) * max(0.0, cocoa_price_usd)
