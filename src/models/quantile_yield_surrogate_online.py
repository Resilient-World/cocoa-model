"""
Online conformal calibration for :class:`~models.cqr.QuantileYieldSurrogate`.

Swaps static split-conformal ``Q_hat`` for adaptive ACI / PID / ECI updaters under
CMIP6-style distribution shift (e.g. ``/simulate-scenario``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from models.aci import DEFAULT_SCENARIO_HORIZONS, AdaptiveConformalInference, MultiStepACI
from models.conformal_pid import ConformalPID
from models.cqr import CQRInterval, ConformalCalibrator, QuantileYieldSurrogate
from models.eci import ECICutoff, ECIIntegral, ErrorQuantifiedConformalInference
from models.online_conformal_base import conformal_quantile

OnlineMethod = Literal["aci", "conformal_pid", "eci", "eci_cutoff", "eci_integral"]

_UPDATER_TYPES = (
    AdaptiveConformalInference,
    ConformalPID,
    ErrorQuantifiedConformalInference,
    ECICutoff,
    ECIIntegral,
)


def _build_updater(
    method: OnlineMethod,
    alpha: float,
    **kwargs: Any,
) -> AdaptiveConformalInference | ConformalPID | ErrorQuantifiedConformalInference | ECICutoff | ECIIntegral:
    if method == "aci":
        return AdaptiveConformalInference(
            alpha,
            eta=float(kwargs.get("eta", 0.005)),
            q_init=float(kwargs.get("q_init", 0.0)),
        )
    if method == "conformal_pid":
        return ConformalPID(
            alpha,
            eta=float(kwargs.get("eta", 0.01)),
            window=int(kwargs.get("window", 100)),
            q_init=float(kwargs.get("q_init", 0.0)),
            g_prime=kwargs.get("g_prime"),
        )
    if method == "eci":
        return ErrorQuantifiedConformalInference(
            alpha,
            eta=float(kwargs.get("eta", 0.01)),
            c=float(kwargs.get("c", 1.0)),
            window=int(kwargs.get("window", 100)),
            q_init=float(kwargs.get("q_init", 0.0)),
        )
    if method == "eci_cutoff":
        return ECICutoff(
            alpha,
            eta=float(kwargs.get("eta", 0.01)),
            c=float(kwargs.get("c", 1.0)),
            h=float(kwargs.get("h", 0.5)),
            window=int(kwargs.get("window", 100)),
            q_init=float(kwargs.get("q_init", 0.0)),
        )
    if method == "eci_integral":
        return ECIIntegral(
            alpha,
            eta=float(kwargs.get("eta", 0.01)),
            c=float(kwargs.get("c", 1.0)),
            decay=float(kwargs.get("decay", 0.95)),
            window=int(kwargs.get("window", 100)),
            q_init=float(kwargs.get("q_init", 0.0)),
        )
    raise ValueError(f"Unknown online_method: {method}")


class QuantileYieldSurrogateOnline:
    """
    Wraps a trained :class:`QuantileYieldSurrogate` with an online conformal threshold ``q_t``.

    When ``observed_y`` is supplied, conformity scores update ``q_t`` before forming intervals.
    """

    def __init__(
        self,
        model: QuantileYieldSurrogate,
        *,
        online_method: OnlineMethod = "eci",
        alpha: float = 0.1,
        warm_start_scores: np.ndarray | None = None,
        **method_kwargs: Any,
    ) -> None:
        self.model = model
        self.online_method = online_method
        self.alpha = float(alpha)
        self.updater = _build_updater(online_method, alpha, **method_kwargs)
        if warm_start_scores is not None and len(warm_start_scores) > 0:
            q0 = conformal_quantile(np.asarray(warm_start_scores), alpha)
            if hasattr(self.updater, "q"):
                self.updater.q = q0

    @property
    def current_threshold(self) -> float:
        return self.updater.current_threshold

    @torch.no_grad()
    def predict_with_online_calibration(
        self,
        X: tuple[Tensor, Tensor],
        observed_y: float | np.ndarray | Tensor | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> CQRInterval:
        """
        Predict conformalized quantile interval using the current online threshold.

        If ``observed_y`` is set, the conformity score is computed and the updater
        is stepped **before** returning the interval (online learning step).
        """
        self.model.eval()
        climate, static = X
        dev = torch.device(device)
        q_pred = self.model(climate.to(dev), static.to(dev)).detach().cpu().numpy()
        q_lo, q_med, q_hi = float(q_pred[0, 0]), float(q_pred[0, 1]), float(q_pred[0, 2])

        if observed_y is not None:
            y_val = float(
                observed_y.item()
                if torch.is_tensor(observed_y)
                else np.asarray(observed_y, dtype=np.float64).reshape(-1)[0]
            )
            score = float(
                ConformalCalibrator.conformity_scores(
                    np.array([y_val]),
                    np.array([q_lo]),
                    np.array([q_hi]),
                )[0]
            )
            self.updater.update(score)

        q_adj = self.current_threshold
        return CQRInterval(
            lower=q_lo - q_adj,
            median=q_med,
            upper=q_hi + q_adj,
            q_adjustment=q_adj,
        )


class HorizonOnlineCalibrator:
    """
    Lightweight adapter: one ``MultiStepACI`` threshold per scenario horizon.

    Used with a single shared quantile model; horizons map to independent ``q_h``.
    """

    def __init__(
        self,
        multistep: MultiStepACI,
        *,
        q_lo: float = 0.0,
        q_hi: float = 1.0,
    ) -> None:
        self.multistep = multistep
        self.q_lo = q_lo
        self.q_hi = q_hi

    def update_horizon(
        self,
        horizon: str,
        observed_y: float,
        q_lo: float,
        q_hi: float,
    ) -> float:
        score = float(
            ConformalCalibrator.conformity_scores(
                np.array([observed_y]),
                np.array([q_lo]),
                np.array([q_hi]),
            )[0]
        )
        covered = observed_y >= q_lo - self.multistep.threshold(horizon) and observed_y <= q_hi + self.multistep.threshold(horizon)
        thresholds = self.multistep.update(
            np.array([score]),
            np.array([covered]),
        )
        return thresholds[horizon]

    def interval_for_horizon(
        self,
        horizon: str,
        q_lo: float,
        q_med: float,
        q_hi: float,
    ) -> CQRInterval:
        q_adj = self.multistep.threshold(horizon)
        return CQRInterval(
            lower=q_lo - q_adj,
            median=q_med,
            upper=q_hi + q_adj,
            q_adjustment=q_adj,
        )


def factory_multi_horizon(
    model: QuantileYieldSurrogate,
    *,
    online_method: OnlineMethod = "eci",
    alpha: float = 0.1,
    horizons: list[str] | None = None,
    **method_kwargs: Any,
) -> dict[str, QuantileYieldSurrogateOnline]:
    """
    One :class:`QuantileYieldSurrogateOnline` per scenario horizon (shared weights, separate ACI state).

    Implemented via independent updaters (equivalent to :class:`MultiStepACI` stratification).
    """
    hz = horizons or list(DEFAULT_SCENARIO_HORIZONS)
    return {
        h: QuantileYieldSurrogateOnline(
            model,
            online_method=online_method,
            alpha=alpha,
            **method_kwargs,
        )
        for h in hz
    }


__all__ = [
    "HorizonOnlineCalibrator",
    "OnlineMethod",
    "QuantileYieldSurrogateOnline",
    "factory_multi_horizon",
]
