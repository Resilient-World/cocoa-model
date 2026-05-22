"""
Parametric cocoa yield insurance, portfolio VaR, and reinsurance layering.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

CopulaKind = Literal["gaussian"]


def parametric_payout(
    observed_yield: float,
    strike_yield: float,
    sum_insured: float,
    *,
    basis_risk_factor: float = 0.15,
) -> float:
    """
    Index-triggered payout on yield shortfall.

    Payout = sum_insured × max(0, (strike − observed) / strike) × (1 − basis_risk).
    """
    if strike_yield <= 0 or sum_insured <= 0:
        return 0.0
    shortfall = max(0.0, (strike_yield - observed_yield) / strike_yield)
    haircut = max(0.0, min(1.0, 1.0 - basis_risk_factor))
    return float(sum_insured * shortfall * haircut)


def _norm_ppf(p: float) -> float:
    """Inverse CDF for standard normal (Acklam approximation)."""
    p = max(1e-9, min(1.0 - 1e-9, p))
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479824614460e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989775598239e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.224676258516688e-01,
        -2.400758227987018e00,
        -2.267715495055226e00,
        3.224676258516688e-01,
        7.784894002430293e-03,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def aggregate_portfolio_var(
    farm_payouts: np.ndarray,
    *,
    copula: CopulaKind = "gaussian",
    rho: float = 0.4,
    alpha: float = 0.99,
) -> float:
    """
    Portfolio value-at-risk for correlated parametric payouts (Gaussian copula).

    Equal-weight farms; variance = rho·σ² + (1−ρ)·σ²/n with σ = mean payout scale.
    """
    payouts = np.asarray(farm_payouts, dtype=np.float64).reshape(-1)
    n = payouts.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(payouts[0])

    sigma = float(np.std(payouts, ddof=1)) if n > 1 else float(payouts[0])
    mean = float(np.mean(payouts))
    rho = max(0.0, min(0.99, rho))

    if copula != "gaussian":
        raise ValueError(f"Unsupported copula: {copula}")

    # Variance of sum under equicorrelation: Var(S) = n·σ² + n(n−1)·ρ·σ²
    var_sum = (n + n * (n - 1) * rho) * sigma**2
    std_sum = math.sqrt(max(var_sum, 1e-12))
    z = _norm_ppf(alpha)
    return float(mean * n + z * std_sum)


def reinsurance_layer(
    losses: np.ndarray | float,
    attachment: float,
    exhaust: float,
) -> dict[str, float]:
    """
    Single excess-of-loss layer: ceded losses between attachment and exhaust.

    Returns gross, ceded, and net retained amounts.
    """
    if exhaust <= attachment:
        raise ValueError("exhaust must exceed attachment")

    if np.isscalar(losses):
        gross = float(losses)
    else:
        gross = float(np.sum(np.asarray(losses, dtype=np.float64)))

    excess = max(0.0, gross - attachment)
    capacity = exhaust - attachment
    ceded = min(excess, capacity)
    net = gross - ceded
    return {
        "gross": gross,
        "ceded": ceded,
        "net": net,
        "attachment": attachment,
        "exhaust": exhaust,
    }
