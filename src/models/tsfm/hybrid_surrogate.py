"""
Hybrid yield surrogate: blends TSFM ensemble with CASEJ for scenario simulation.

- Uses CASEJ surrogate for CO2-physiology under SSP scenarios (existing)
- Uses TSFM ensemble for monthly yield trajectory forecasting
- Blends via stratum-fitted weights, defaulting to 60% TSFM / 40% CASEJ
- Falls back to 100% CASEJ for SSP scenarios outside the CMIP6 ensemble envelope
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from torch import Tensor

from models.process.casej_surrogate import CASEJSurrogate
from models.tsfm.ensemble import TsfmEnsemble
from models.tsfm.wrappers import TsfmForecast

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TSFM_WEIGHT = 0.6
DEFAULT_CASEJ_WEIGHT = 0.4

# CMIP6 CO2 envelope (ppm) — outside this range, fall back to 100% CASEJ.
CMIP6_CO2_MIN = 350.0
CMIP6_CO2_MAX = 950.0


class HybridYieldSurrogate:
    """
    Blended yield surrogate for scenario simulation.

    Combines:
    - :class:`~models.process.casej_surrogate.CASEJSurrogate` for CO2 physiology
    - :class:`~models.tsfm.ensemble.TsfmEnsemble` for monthly trajectory forecasting

    Blend weights are stratum-fitted; defaults to 60% TSFM / 40% CASEJ.
    Falls back to 100% CASEJ when CO2 is outside the CMIP6 ensemble envelope.
    """

    def __init__(
        self,
        casej_model: CASEJSurrogate,
        *,
        tsfm_ensemble: TsfmEnsemble | None = None,
        tsfm_weight: float = DEFAULT_TSFM_WEIGHT,
        casej_weight: float = DEFAULT_CASEJ_WEIGHT,
        device: str | None = None,
    ) -> None:
        self.casej_model = casej_model
        self.tsfm_ensemble = tsfm_ensemble or TsfmEnsemble(device=device)
        self.tsfm_weight = tsfm_weight
        self.casej_weight = casej_weight
        self.device = device

        self._enabled = os.environ.get("TSFM_ENABLED", "false").lower() in ("1", "true", "yes")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _co2_in_envelope(self, co2_ppm: float) -> bool:
        """Check whether CO2 is within the CMIP6 ensemble training envelope."""
        return CMIP6_CO2_MIN <= co2_ppm <= CMIP6_CO2_MAX

    def _should_use_tsfm(self, co2_ppm: float) -> bool:
        """Determine whether TSFM ensemble should contribute to the blend."""
        if not self._enabled:
            return False
        return self._co2_in_envelope(co2_ppm)

    @torch.no_grad()
    def forward(
        self,
        climate: Tensor,
        static: Tensor,
        co2_ppm: Tensor | None = None,
        *,
        monthly_history: np.ndarray | None = None,
        horizon: int = 12,
        region: str = "default",
    ) -> dict[str, Any]:
        """
        Produce blended yield forecast.

        Parameters
        ----------
        climate:
            Daily climate ``[B, 365, C]`` for CASEJ.
        static:
            Site static ``[B, F]`` for CASEJ.
        co2_ppm:
            CO2 concentration override ``[B]``.
        monthly_history:
            ``[time_steps, features]`` multivariate history for TSFM ensemble.
        horizon:
            Forecast horizon for TSFM (months).
        region:
            Region key for per-region TSFM weights.

        Returns
        -------
        dict with ``casej_mean``, ``tsfm_forecast``, ``blended_mean``, ``blend_weights``,
        ``tsfm_active``.
        """
        co2_val = float(co2_ppm.mean().item()) if co2_ppm is not None else 420.0

        casej_pred = self.casej_model(climate, static, co2_ppm=co2_ppm)
        casej_mean = float(casej_pred.mean().item())

        tsfm_active = self._should_use_tsfm(co2_val)
        tsfm_forecast: TsfmForecast | None = None
        tsfm_mean: float | None = None

        if tsfm_active and monthly_history is not None and monthly_history.size > 0:
            try:
                tsfm_forecast = self.tsfm_ensemble.forecast(
                    monthly_history,
                    horizon=horizon,
                    region=region,
                )
                tsfm_mean = float(tsfm_forecast.p50.mean())
            except Exception as exc:
                log.warning("TSFM ensemble forecast failed, falling back to CASEJ only", error=str(exc))
                tsfm_active = False

        if tsfm_active and tsfm_mean is not None:
            blended = self.tsfm_weight * tsfm_mean + self.casej_weight * casej_mean
            blend_weights = {"tsfm": self.tsfm_weight, "casej": self.casej_weight}
        else:
            blended = casej_mean
            blend_weights = {"tsfm": 0.0, "casej": 1.0}
            tsfm_active = False

        return {
            "casej_mean": casej_mean,
            "tsfm_forecast": tsfm_forecast,
            "tsfm_mean": tsfm_mean,
            "blended_mean": blended,
            "blend_weights": blend_weights,
            "tsfm_active": tsfm_active,
        }

    def predict_scenario(
        self,
        climate: Tensor,
        static: Tensor,
        co2_ppm: Tensor,
        *,
        monthly_history: np.ndarray | None = None,
        horizon: int = 12,
        region: str = "default",
    ) -> dict[str, Any]:
        """Alias for :meth:`forward` matching scenario API conventions."""
        return self.forward(
            climate,
            static,
            co2_ppm=co2_ppm,
            monthly_history=monthly_history,
            horizon=horizon,
            region=region,
        )
