"""Parametric insurance pricing math for cocoa avoided-loss products."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from analysis.dvds import MarginalSensitivityModel

DVDS_COOPERATIVE_LAMBDA = 1.5
DEFAULT_PRICE_USD_PER_T = 3000.0


@dataclass(frozen=True)
class BasisRiskReport:
    correlation: float
    rmse: float
    regression_slope: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "correlation": self.correlation,
            "rmse": self.rmse,
            "regression_slope": self.regression_slope,
            "r2": self.r2,
        }


@dataclass(frozen=True)
class ParametricPricingReport:
    fair_premium_usd: float
    loaded_premium_usd: float
    expected_payout_usd: float
    basis_risk_r2: float
    lambda_sensitivity: dict[str, float]
    conformal_adjusted_volatility: float
    strike_t_per_ha: float

    def as_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "fair_premium_usd": self.fair_premium_usd,
            "loaded_premium_usd": self.loaded_premium_usd,
            "expected_payout_usd": self.expected_payout_usd,
            "basis_risk_r2": self.basis_risk_r2,
            "lambda_sensitivity": self.lambda_sensitivity,
            "conformal_adjusted_volatility": self.conformal_adjusted_volatility,
            "strike_t_per_ha": self.strike_t_per_ha,
        }


def _as_1d(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    return arr


def compute_basis_risk(
    realized_loss_t: np.ndarray | list[float] | tuple[float, ...],
    parametric_payout_t: np.ndarray | list[float] | tuple[float, ...],
) -> BasisRiskReport:
    """Return correlation, RMSE, slope, and R² for realized loss vs index payout."""
    realized = _as_1d(realized_loss_t)
    payout = _as_1d(parametric_payout_t)
    if realized.shape != payout.shape:
        raise ValueError("realized_loss_t and parametric_payout_t must have the same shape")
    diff = realized - payout
    rmse = float(np.sqrt(np.mean(diff**2)))
    realized_var = float(np.var(realized))
    payout_var = float(np.var(payout))
    if realized_var <= 1e-12 or payout_var <= 1e-12:
        correlation = 1.0 if rmse <= 1e-12 else 0.0
    else:
        correlation = float(np.corrcoef(realized, payout)[0, 1])
    cov = float(np.mean((realized - realized.mean()) * (payout - payout.mean())))
    slope = 0.0 if payout_var <= 1e-12 else float(cov / payout_var)
    ss_tot = float(np.sum((realized - realized.mean()) ** 2))
    ss_res = float(np.sum(diff**2))
    r2 = 1.0 if ss_tot <= 1e-12 and ss_res <= 1e-12 else max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))
    return BasisRiskReport(correlation=correlation, rmse=rmse, regression_slope=slope, r2=float(r2))


def _conformal_width(conformal_bounds: Mapping[str, float] | tuple[float, float] | None) -> float:
    if conformal_bounds is None:
        return 0.0
    if isinstance(conformal_bounds, tuple):
        lower, upper = conformal_bounds
        return max(0.0, float(upper) - float(lower))
    lower = float(conformal_bounds.get("lower", conformal_bounds.get("ci_low", 0.0)))
    upper = float(conformal_bounds.get("upper", conformal_bounds.get("ci_high", lower)))
    return max(0.0, upper - lower)


def price_parametric_trigger(
    strike_t_per_ha: float,
    scenario_samples: np.ndarray | list[float] | tuple[float, ...],
    conformal_bounds: Mapping[str, float] | tuple[float, float] | None,
    *,
    farm_size_ha: float = 1.0,
    price_usd_per_t: float = DEFAULT_PRICE_USD_PER_T,
    realized_loss_t: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lambda_: float = DVDS_COOPERATIVE_LAMBDA,
) -> ParametricPricingReport:
    """
    Price an index trigger by Monte Carlo expected payout plus DVDS/conformal loading.

    ``scenario_samples`` are yield samples in t/ha. Payout is the strike shortfall
    converted to tonnes by farm area and then to USD by cocoa price.
    """
    if strike_t_per_ha <= 0.0:
        raise ValueError("strike_t_per_ha must be positive")
    samples = _as_1d(scenario_samples)
    shortfall_t_per_ha = np.maximum(0.0, strike_t_per_ha - samples)
    payout_t = shortfall_t_per_ha * max(0.0, farm_size_ha)
    payout_usd = payout_t * max(0.0, price_usd_per_t)
    expected = float(np.mean(payout_usd))
    volatility = float(np.std(payout_usd, ddof=1)) if payout_usd.size > 1 else 0.0
    conformal_vol = (
        volatility + _conformal_width(conformal_bounds) * farm_size_ha * price_usd_per_t / 3.29
    )
    msm = MarginalSensitivityModel(lambda_)
    tail_loading = (msm.lambda_ - 1.0) / msm.lambda_ * conformal_vol
    loaded = expected + tail_loading

    realized = _as_1d(realized_loss_t) if realized_loss_t is not None else payout_t
    basis = compute_basis_risk(realized, payout_t)
    sensitivity = {
        "lambda_1_0": expected,
        f"lambda_{lambda_:.1f}": loaded,
        "lambda_2_0": expected + 0.5 * conformal_vol,
    }
    return ParametricPricingReport(
        fair_premium_usd=expected,
        loaded_premium_usd=float(loaded),
        expected_payout_usd=expected,
        basis_risk_r2=basis.r2,
        lambda_sensitivity=sensitivity,
        conformal_adjusted_volatility=float(conformal_vol),
        strike_t_per_ha=float(strike_t_per_ha),
    )


def smile_corrected_pricing(
    strikes: np.ndarray | list[float] | tuple[float, ...],
    base_vol: float,
    smile_curve: Mapping[float, float] | np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Apply out-of-the-money volatility smile multipliers to strike grid."""
    strike_arr = _as_1d(strikes)
    if isinstance(smile_curve, Mapping):
        knots = np.array(sorted(float(k) for k in smile_curve), dtype=np.float64)
        vals = np.array([float(smile_curve[float(k)]) for k in knots], dtype=np.float64)
        multipliers = np.interp(strike_arr, knots, vals, left=vals[0], right=vals[-1])
    else:
        multipliers = _as_1d(smile_curve)
        if multipliers.size != strike_arr.size:
            raise ValueError("smile_curve array must match strikes length")
    corrected = np.maximum(0.0, base_vol) * np.maximum(0.0, multipliers)
    return np.asarray(corrected, dtype=np.float64)


__all__ = [
    "DVDS_COOPERATIVE_LAMBDA",
    "BasisRiskReport",
    "ParametricPricingReport",
    "compute_basis_risk",
    "price_parametric_trigger",
    "smile_corrected_pricing",
]
