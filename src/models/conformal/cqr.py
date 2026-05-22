"""
Conformalized Quantile Regression (CQR) for cocoa yield uncertainty.

Quantile neural head (0.05 / 0.5 / 0.95) plus split-conformal calibration gives
valid marginal coverage for insurance and parametric-payout use cases, replacing
under-covering Monte Carlo dropout intervals (~40% empirical at 80% nominal).
"""

from __future__ import annotations

import structlog

from pathlib import Path
from typing import Any, NamedTuple, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.surrogate.yield_surrogate import (
    MCDropout,
    MechanisticCore,
    N_CLIMATE_CHANNELS,
    N_STATIC_SITE,
    YieldSurrogateModel,
)

log = structlog.get_logger(__name__)

DEFAULT_QUANTILES: tuple[float, float, float] = (0.05, 0.5, 0.95)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CQR_CHECKPOINT = _REPO_ROOT / "models" / "cqr_yield.pt"
DEFAULT_CQR_CALIBRATOR = _REPO_ROOT / "models" / "cqr_calibrator.joblib"


class CQRInterval(NamedTuple):
    """Conformalized yield interval (tonnes/ha)."""

    lower: float
    median: float
    upper: float
    q_adjustment: float


class QuantilePrediction(NamedTuple):
    """Raw quantile head before conformal adjustment."""

    q_lo: Tensor
    q_med: Tensor
    q_hi: Tensor


def pinball_loss(
    y_pred: Tensor,
    y_true: Tensor,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> Tensor:
    """
    Pinball (quantile) loss averaged over quantile dimensions.

    Parameters
    ----------
    y_pred:
        ``[B, Q]`` predicted quantiles matching ``quantiles`` order.
    y_true:
        ``[B]`` or ``[B, 1]`` targets.
    """
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred must be [B, Q], got {tuple(y_pred.shape)}")
    q_count = y_pred.shape[1]
    if len(quantiles) != q_count:
        raise ValueError(f"len(quantiles)={len(quantiles)} != y_pred width {q_count}")

    target = y_true.view(-1, 1).expand_as(y_pred)
    losses: list[Tensor] = []
    for j, tau in enumerate(quantiles):
        err = target[:, j] - y_pred[:, j]
        losses.append(torch.maximum(tau * err, (tau - 1.0) * err))
    return torch.stack(losses, dim=1).mean()


class QuantileYieldSurrogate(nn.Module):
    """
    Same mechanistic + GRU/MLP backbone as :class:`~models.surrogate.yield_surrogate.YieldSurrogateModel`,
    with a three-output head for quantiles ``[0.05, 0.5, 0.95]`` (tonnes/ha).
    """

    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES

    def __init__(
        self,
        sequence_length: int = 365,
        climate_features: int = N_CLIMATE_CHANNELS,
        static_features: int = N_STATIC_SITE,
        galileo_dim: int = 0,
        static_hidden: int = 64,
        head_hidden: int = 64,
        gru_hidden: int = 96,
        gru_layers: int = 2,
        attn_heads: int = 4,
        dropout: float = 0.1,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        super().__init__()
        self._base = YieldSurrogateModel(
            sequence_length=sequence_length,
            climate_features=climate_features,
            static_features=static_features,
            galileo_dim=galileo_dim,
            static_hidden=static_hidden,
            head_hidden=head_hidden,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            attn_heads=attn_heads,
            dropout=dropout,
        )
        self.quantiles = tuple(float(q) for q in quantiles)
        if len(self.quantiles) != 3:
            raise ValueError("QuantileYieldSurrogate expects exactly three quantiles")

        fusion_in = self._base.residual_head[0].in_features  # type: ignore[index]
        self.quantile_head = nn.Sequential(
            nn.Linear(fusion_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, 3),
        )
        nn.init.normal_(self.quantile_head[-1].weight, std=0.01)  # type: ignore[index]
        if self.quantile_head[-1].bias is not None:  # type: ignore[index]
            nn.init.zeros_(self.quantile_head[-1].bias)  # type: ignore[index]
        self._replace_mc_dropout_with_standard()

    def _replace_mc_dropout_with_standard(self) -> None:
        """Use deterministic dropout-off inference (CQR supplies uncertainty)."""

        def _walk(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, MCDropout):
                    setattr(module, name, nn.Dropout(p=child.p))
                else:
                    _walk(child)

        _walk(self._base)

    @property
    def mechanistic(self) -> MechanisticCore:
        return self._base.mechanistic

    def _fusion_features(
        self,
        climate: Tensor,
        static: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._base._validate_inputs(climate, static)
        climate_full = self._base._pad_climate(climate)
        traces = self._base.mechanistic(climate_full, static)
        gru_out, _ = self._base.climate_gru(climate_full)
        climate_emb = self._base.attention_pool(gru_out)
        static_emb = self._base.static_mlp(static)
        stress_summary = traces["stress_trace"].mean(dim=1)
        fused = torch.cat([climate_emb, static_emb, stress_summary], dim=1)
        return fused, traces["y_mech"]

    def forward(self, climate: Tensor, static: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            ``[B, 3]`` quantiles (q05, q50, q95) in tonnes/ha.
        """
        fused, y_mech = self._fusion_features(climate, static)
        residual_q = self.quantile_head(fused)
        return y_mech.unsqueeze(1) + residual_q

    def forward_quantiles(self, climate: Tensor, static: Tensor) -> QuantilePrediction:
        q = self.forward(climate, static)
        return QuantilePrediction(q_lo=q[:, 0], q_med=q[:, 1], q_hi=q[:, 2])

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:  # noqa: D102
        base_sd = self._base.state_dict(*args, **kwargs)
        head_sd = {f"quantile_head.{k}": v for k, v in self.quantile_head.state_dict().items()}
        return {**base_sd, **head_sd}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:  # noqa: D102
        head_keys = {k for k in state_dict if k.startswith("quantile_head.")}
        base_dict = {k: v for k, v in state_dict.items() if k not in head_keys}
        head_dict = {k.replace("quantile_head.", "", 1): v for k, v in state_dict.items() if k in head_keys}
        self._base.load_state_dict(base_dict, strict=False)
        if head_dict:
            self.quantile_head.load_state_dict(head_dict, strict=strict)


class ConformalCalibrator:
    """
    Split-conformal adjustment for quantile regression (Romano et al., 2019).

    Conformity score per calibration point::

        E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))

    Stored adjustment ``Q`` is the ``(1 - alpha)`` quantile of ``{E_i}``.
    """

    def __init__(self) -> None:
        self.alpha: float = 0.1
        self.Q_hat: float | None = None
        self.quantiles: tuple[float, float, float] = DEFAULT_QUANTILES
        self.n_calibration: int = 0
        self.empirical_coverage: float | None = None
        self.cv_strategy: str | None = None
        self.fold_coverages: list[float] = []
        self.recommended_block_km: float | None = None

    @staticmethod
    def conformity_scores(
        y_true: np.ndarray,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
    ) -> np.ndarray:
        """``E_i = max(q_lo - y, y - q_hi)``."""
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)
        lo = np.asarray(q_lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(q_hi, dtype=np.float64).reshape(-1)
        return np.maximum(lo - y, y - hi)

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """Finite-sample conformal quantile (Romano et al., 2019)."""
        scores = np.sort(np.asarray(scores, dtype=np.float64))
        n = len(scores)
        if n == 0:
            return 0.0
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        return float(scores[k - 1])

    @torch.no_grad()
    def fit(
        self,
        model: QuantileYieldSurrogate,
        X_cal: tuple[Tensor, Tensor],
        y_cal: Tensor | np.ndarray,
        *,
        alpha: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> ConformalCalibrator:
        """
        Calibrate conformal adjustment on held-out split.

        Parameters
        ----------
        X_cal:
            ``(climate, static)`` tensors with batch = calibration size.
        y_cal:
            Observed yields ``[N]`` (tonnes/ha).
        """
        model.eval()
        climate, static = X_cal
        dev = torch.device(device)
        climate = climate.to(dev)
        static = static.to(dev)
        y_np = np.asarray(
            y_cal.detach().cpu().numpy() if torch.is_tensor(y_cal) else y_cal,
            dtype=np.float64,
        ).reshape(-1)

        q = model(climate, static).detach().cpu().numpy()
        q_lo, q_hi = q[:, 0], q[:, 2]
        scores = self.conformity_scores(y_np, q_lo, q_hi)
        self.alpha = float(alpha)
        self.Q_hat = self._conformal_quantile(scores, alpha)
        self.quantiles = model.quantiles
        self.n_calibration = len(y_np)
        covered = (y_np >= q_lo - self.Q_hat) & (y_np <= q_hi + self.Q_hat)
        self.empirical_coverage = float(covered.mean())
        log.info(
            "CQR calibrator fit n=%d alpha=%.2f Q=%.4f cal_coverage=%.3f",
            self.n_calibration,
            self.alpha,
            self.Q_hat,
            self.empirical_coverage,
        )
        return self

    @torch.no_grad()
    def fit_blocked(
        self,
        model: QuantileYieldSurrogate,
        X: tuple[Tensor, Tensor],
        y: Tensor | np.ndarray,
        splitter: object,
        *,
        coords: tuple[np.ndarray, np.ndarray] | None = None,
        years: np.ndarray | None = None,
        alpha: float = 0.1,
        fold: int | None = None,
        device: torch.device | str = "cpu",
    ) -> ConformalCalibrator:
        """
        Calibrate on spatially or temporally held-out folds (not i.i.d. random split).

        Test-fold indices supply calibration scores; ``Q_hat`` is the median across folds
        unless ``fold`` selects one fold only.
        """
        from validation.spatial_cv import BufferedLOO, SpatialBlockSplit
        from validation.temporal_cv import ForwardChainSplit

        climate, static = X
        y_np = np.asarray(
            y.detach().cpu().numpy() if torch.is_tensor(y) else y,
            dtype=np.float64,
        ).reshape(-1)
        n = len(y_np)

        if isinstance(splitter, ForwardChainSplit):
            if years is None:
                raise ValueError("years required for ForwardChainSplit calibration")
            splits = list(splitter.split(years))
            self.cv_strategy = "temporal_forward"
        elif isinstance(splitter, (SpatialBlockSplit, BufferedLOO)):
            if coords is None:
                raise ValueError("coords (lat, lon) required for spatial calibration")
            lats, lons = coords
            splits = list(splitter.split(lats, lons))
            self.cv_strategy = (
                "buffered_loo" if isinstance(splitter, BufferedLOO) else "spatial_block"
            )
            if isinstance(splitter, SpatialBlockSplit):
                self.recommended_block_km = splitter.block_size_km
        else:
            raise TypeError(f"Unsupported splitter type: {type(splitter)}")

        q_hats: list[float] = []
        coverages: list[float] = []
        dev = torch.device(device)

        for fold_i, (_train_idx, test_idx) in enumerate(splits):
            if fold is not None and fold_i != fold:
                continue
            if len(test_idx) == 0:
                continue
            cal_climate = climate[test_idx].to(dev)
            cal_static = static[test_idx].to(dev)
            y_cal = y_np[test_idx]
            q = model(cal_climate, cal_static).detach().cpu().numpy()
            scores = self.conformity_scores(y_cal, q[:, 0], q[:, 2])
            qh = self._conformal_quantile(scores, alpha)
            q_hats.append(qh)
            lo, hi = q[:, 0] - qh, q[:, 2] + qh
            coverages.append(float(np.mean((y_cal >= lo) & (y_cal <= hi))))

        if not q_hats:
            raise RuntimeError("fit_blocked produced no valid folds")

        self.alpha = float(alpha)
        self.Q_hat = float(np.median(q_hats))
        self.fold_coverages = coverages
        self.n_calibration = n
        self.empirical_coverage = float(np.median(coverages))
        log.info(
            "CQR fit_blocked strategy=%s folds=%d Q_median=%.4f coverage_median=%.3f",
            self.cv_strategy,
            len(q_hats),
            self.Q_hat,
            self.empirical_coverage,
        )
        return self

    def empirical_coverage_on(
        self,
        y_true: np.ndarray,
        lowers: np.ndarray,
        uppers: np.ndarray,
    ) -> float:
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)
        lo = np.asarray(lowers, dtype=np.float64).reshape(-1)
        hi = np.asarray(uppers, dtype=np.float64).reshape(-1)
        return float(np.mean((y >= lo) & (y <= hi)))

    @torch.no_grad()
    def predict_interval(
        self,
        model: QuantileYieldSurrogate,
        X: tuple[Tensor, Tensor],
        *,
        device: torch.device | str = "cpu",
    ) -> CQRInterval:
        """Return conformalized (lower, median, upper) for a single batch (often B=1)."""
        if self.Q_hat is None:
            raise RuntimeError("ConformalCalibrator.fit must be called before predict_interval")

        model.eval()
        climate, static = X
        dev = torch.device(device)
        q = model(climate.to(dev), static.to(dev)).detach().cpu().numpy()
        q_lo, q_med, q_hi = float(q[0, 0]), float(q[0, 1]), float(q[0, 2])
        q_adj = float(self.Q_hat)
        return CQRInterval(
            lower=q_lo - q_adj,
            median=q_med,
            upper=q_hi + q_adj,
            q_adjustment=q_adj,
        )

    def predict_interval_batch(
        self,
        model: QuantileYieldSurrogate,
        X: tuple[Tensor, Tensor],
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized conformal intervals; returns (lower, median, upper) arrays."""
        if self.Q_hat is None:
            raise RuntimeError("ConformalCalibrator.fit must be called before predict_interval")

        model.eval()
        climate, static = X
        dev = torch.device(device)
        q = model(climate.to(dev), static.to(dev)).detach().cpu().numpy()
        q_adj = float(self.Q_hat)
        return q[:, 0] - q_adj, q[:, 1], q[:, 2] + q_adj

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "Q_hat": self.Q_hat,
            "quantiles": self.quantiles,
            "n_calibration": self.n_calibration,
            "empirical_coverage": self.empirical_coverage,
            "cv_strategy": self.cv_strategy,
            "fold_coverages": self.fold_coverages,
            "recommended_block_km": self.recommended_block_km,
        }
        joblib.dump(payload, path)
        log.info("Saved CQR calibrator to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> ConformalCalibrator:
        payload = joblib.load(path)
        obj = cls()
        obj.alpha = float(payload["alpha"])
        obj.Q_hat = payload.get("Q_hat")
        obj.quantiles = tuple(payload.get("quantiles", DEFAULT_QUANTILES))
        obj.n_calibration = int(payload.get("n_calibration", 0))
        obj.empirical_coverage = payload.get("empirical_coverage")
        obj.cv_strategy = payload.get("cv_strategy")
        obj.fold_coverages = list(payload.get("fold_coverages") or [])
        rb = payload.get("recommended_block_km")
        obj.recommended_block_km = float(rb) if rb is not None else None
        return obj


def load_quantile_yield_model(
    checkpoint_path: str | Path | None = None,
    *,
    galileo_dim: int = 0,
    device: str | torch.device = "cpu",
) -> QuantileYieldSurrogate:
    """Load :class:`QuantileYieldSurrogate` weights from ``cqr_yield.pt``."""
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CQR_CHECKPOINT
    model = QuantileYieldSurrogate(galileo_dim=galileo_dim)
    if path.is_file():
        state = torch.load(path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        log.info("Loaded QuantileYieldSurrogate from %s", path)
    else:
        log.warning("CQR checkpoint missing at %s; using random weights", path)
    model.to(device)
    model.eval()
    return model


def load_cqr_calibrator(path: str | Path | None = None) -> ConformalCalibrator | None:
    """Load calibrator if present."""
    p = Path(path) if path else DEFAULT_CQR_CALIBRATOR
    if not p.is_file():
        return None
    return ConformalCalibrator.load(p)


__all__ = [
    "CQRInterval",
    "ConformalCalibrator",
    "DEFAULT_CQR_CALIBRATOR",
    "DEFAULT_CQR_CHECKPOINT",
    "DEFAULT_QUANTILES",
    "QuantilePrediction",
    "QuantileYieldSurrogate",
    "load_cqr_calibrator",
    "load_quantile_yield_model",
    "pinball_loss",
]
