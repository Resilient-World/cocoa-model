"""
TSFM conformal integration: stratum key and online conformal wrapper.

Extends the existing :class:`~api.online_conformal_store.OnlineConformalStore`
with TSFM-specific stratum keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from api.online_conformal_store import OnlineConformalStore

log = structlog.get_logger(__name__)


def tsfm_stratum_key(
    scenario: str,
    horizon_year: int | str,
    region: str,
    *,
    downscaling_method: str = "linear_delta",
) -> str:
    """Build TSFM-ensemble conformal stratum key.

    Format: ``{scenario}:{horizon}:{region}:tsfm_ensemble``
    with optional downscaling suffix.
    """
    base = f"{scenario}:{int(horizon_year)}:{region}:tsfm_ensemble"
    if downscaling_method == "corrdiff":
        return f"{base}:corrdiff"
    if downscaling_method == "aurora":
        return f"{base}:aurora"
    return base


class TsfmConformalWrapper:
    """Bridge between TSFM ensemble forecasts and :class:`OnlineConformalStore`.

    Wraps a TSFM ensemble forecast with online conformal threshold adjustment
    using the ECI-Integral updater (default).
    """

    def __init__(
        self,
        store: OnlineConformalStore,
        *,
        alpha: float = 0.1,
    ) -> None:
        self.store = store
        self.alpha = alpha

    def predict_with_conformal(
        self,
        scenario: str,
        horizon_year: int,
        region: str,
        p50: np.ndarray,
        *,
        observed_y: float | None = None,
        downscaling_method: str = "linear_delta",
    ) -> dict[str, Any]:
        """
        Apply online conformal adjustment to TSFM ensemble median forecast.

        Parameters
        ----------
        scenario:
            SSP scenario (``ssp245``, ``ssp585``).
        horizon_year:
            Target year (``2030``, ``2050``, ``2080``).
        region:
            FDP region code.
        p50:
            Ensemble median forecast ``[horizon]``.
        observed_y:
            If provided, update the conformal threshold before returning.
        downscaling_method:
            Downscaling method for key suffix.

        Returns
        -------
        dict with ``adjusted_p50``, ``q_t``, ``coverage_running_avg``, ``covered``.
        """
        key = tsfm_stratum_key(
            scenario, horizon_year, region, downscaling_method=downscaling_method
        )
        updater = self.store.get_updater(key)
        q_t = float(updater.current_threshold)

        covered: bool | None = None
        if observed_y is not None:
            score = float(max(p50[-1] - observed_y, observed_y - p50[-1]))
            covered = float(observed_y - q_t <= p50[-1] <= observed_y + q_t)
            updater.update(score, covered=covered)
            self.store.save_after_update(key, updater, covered=covered)
            q_t = float(updater.current_threshold)

        adjusted = p50.copy()
        cov_avg = self.store.coverage_running_avg(key)

        return {
            "adjusted_p50": adjusted,
            "q_t": q_t,
            "coverage_running_avg": cov_avg,
            "covered": covered,
            "stratum_key": key,
        }
