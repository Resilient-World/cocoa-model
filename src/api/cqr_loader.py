"""Load CQR yield model and conformal calibrator for API inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from models.cqr import (
    ConformalCalibrator,
    QuantileYieldSurrogate,
    load_cqr_calibrator,
    load_quantile_yield_model,
)

if TYPE_CHECKING:
    from api.config import APISettings

logger = logging.getLogger(__name__)


def load_cqr_bundle(
    settings: APISettings | None = None,
) -> tuple[QuantileYieldSurrogate | None, ConformalCalibrator | None]:
    """Return (quantile model, calibrator) when artifacts exist."""
    if settings is None:
        return None, None

    calibrator = load_cqr_calibrator(settings.cqr_calibrator_path)
    if calibrator is None:
        return None, None

    galileo_dim = settings.galileo_embedding_dim if settings.use_galileo_embedding else 0
    model = load_quantile_yield_model(
        settings.cqr_checkpoint_path,
        galileo_dim=galileo_dim,
    )
    return model, calibrator
