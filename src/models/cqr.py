"""Backward-compatible shim; implementation in models.conformal.cqr."""

from __future__ import annotations

from models.conformal.cqr import (
    CQRInterval,
    ConformalCalibrator,
    DEFAULT_CQR_CALIBRATOR,
    DEFAULT_CQR_CHECKPOINT,
    DEFAULT_QUANTILES,
    QuantilePrediction,
    QuantileYieldSurrogate,
    pinball_loss,
)

__all__ = [
    "CQRInterval",
    "ConformalCalibrator",
    "DEFAULT_CQR_CALIBRATOR",
    "DEFAULT_CQR_CHECKPOINT",
    "DEFAULT_QUANTILES",
    "QuantilePrediction",
    "QuantileYieldSurrogate",
    "pinball_loss",
]
