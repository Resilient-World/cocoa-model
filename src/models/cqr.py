"""Backward-compatible shim; implementation in models.conformal.cqr."""

from __future__ import annotations

from models.conformal.cqr import (
    DEFAULT_CQR_CALIBRATOR,
    DEFAULT_CQR_CHECKPOINT,
    DEFAULT_QUANTILES,
    ConformalCalibrator,
    CQRInterval,
    QuantilePrediction,
    QuantileYieldSurrogate,
    load_cqr_calibrator,
    load_quantile_yield_model,
    pinball_loss,
)

__all__ = [
    "DEFAULT_CQR_CALIBRATOR",
    "DEFAULT_CQR_CHECKPOINT",
    "DEFAULT_QUANTILES",
    "CQRInterval",
    "ConformalCalibrator",
    "QuantilePrediction",
    "QuantileYieldSurrogate",
    "load_cqr_calibrator",
    "load_quantile_yield_model",
    "pinball_loss",
]
