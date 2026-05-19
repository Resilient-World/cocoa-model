"""
PINN surrogate for cocoa yield trained on paired CASE2 / ALMANAC outputs, with stacking.

Climate inputs align with ERA5-Land derived features (12 channels × 365 days).
Static inputs cover management and soil (8). Two heads predict log1p yield (kg/ha).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression
from torch import Tensor

from models.yield_surrogate import MCDropout

logger = logging.getLogger(__name__)

CLIMATE_FEATURE_NAMES: tuple[str, ...] = (
    "tmean",
    "tmax",
    "tmin",
    "vpd",
    "et0",
    "cwd",
    "sm_root",
    "precip",
    "srad",
    "gdd_cocoa",
    "heat_days_32c",
    "dry_spell_max",
)
STATIC_FEATURE_NAMES: tuple[str, ...] = (
    "planting_density",
    "tree_age",
    "slai",
    "soil_fc",
    "soil_wp",
    "soil_depth",
    "elevation",
    "latitude",
)

N_CLIMATE = len(CLIMATE_FEATURE_NAMES)
N_STATIC = len(STATIC_FEATURE_NAMES)
SEQ_LEN = 365
CWD_IDX = CLIMATE_FEATURE_NAMES.index("cwd")
HEAT_IDX = CLIMATE_FEATURE_NAMES.index("heat_days_32c")


@dataclass(frozen=True)
class EnsemblePrediction:
    """Calibrated ensemble yield with uncertainty bands."""

    mean: np.ndarray
    std: np.ndarray
    p10: np.ndarray
    p90: np.ndarray


def log1p_yield(y_kg_ha: Tensor) -> Tensor:
    return torch.log1p(torch.clamp(y_kg_ha, min=0.0))


def expm1_yield(y_log: Tensor) -> Tensor:
    return torch.expm1(y_log)


def physics_residual_loss(
    y_hat: Tensor,
    climate: Tensor,
    *,
    lambda_phys: float = 0.05,
    epsilon: float = 1e-3,
) -> Tensor:
    """
    PINN physics penalties on autograd sensitivities w.r.t. climate inputs.

    Sign convention: ``cwd`` is cumulative water deficit (larger = drier). Penalties:

    - ``ReLU(dY/dCWD)``: yield must not increase as deficit grows.
    - ``ReLU(dY/dHeatDays32c)``: yield must not rise with extreme-heat-day exposure.

    (Spec text uses ``ReLU(-dY/dCWD - epsilon)``; with ERA5 ``cwd`` increasing when
    drier, that is equivalent to penalizing ``dY/dCWD > -epsilon``.)
    """
    if not climate.requires_grad:
        climate = climate.detach().requires_grad_(True)

    y_scalar = y_hat.mean(dim=1) if y_hat.ndim == 2 else y_hat
    grads = torch.autograd.grad(
        y_scalar.sum(),
        climate,
        create_graph=True,
        retain_graph=True,
    )[0]
    dy_dcwd = grads[:, CWD_IDX, :].mean(dim=1)
    dy_dheat = grads[:, HEAT_IDX, :].mean(dim=1)

    l_cwd = F.relu(dy_dcwd - epsilon).mean()
    l_heat = F.relu(dy_dheat).mean()
    return lambda_phys * (l_cwd + l_heat)


class CocoaYieldPINN(pl.LightningModule):
    """
    Physics-informed dual-head yield surrogate (CASE2 + ALMANAC targets).

    Parameters
    ----------
    lambda_phys:
        Weight on :func:`physics_residual_loss`.
    lr:
        Adam learning rate.
    dropout:
        MC Dropout probability (active during training and MC inference).
    """

    def __init__(
        self,
        lambda_phys: float = 0.05,
        lr: float = 1e-3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lambda_phys = lambda_phys
        self.lr = lr
        self.dropout_p = dropout

        self.climate_cnn = nn.Sequential(
            nn.Conv1d(N_CLIMATE, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(N_STATIC, 32),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            MCDropout(dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x_climate: Tensor, x_static: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x_climate:
            ``[batch, n_climate, seq_len]`` (default 12 × 365).
        x_static:
            ``[batch, n_static]`` (default 8).

        Returns
        -------
        Tensor
            Log1p yield predictions ``[batch, 2]`` (CASE2, ALMANAC).
        """
        self._validate_inputs(x_climate, x_static)
        c_emb = self.climate_cnn(x_climate).squeeze(-1)  # [B, 64]
        s_emb = self.static_mlp(x_static)  # [B, 32]
        fused = torch.cat([c_emb, s_emb], dim=1)
        return self.fusion_head(fused)

    def _validate_inputs(self, x_climate: Tensor, x_static: Tensor) -> None:
        if x_climate.ndim != 3:
            raise ValueError(f"x_climate must be [B, {N_CLIMATE}, {SEQ_LEN}], got {tuple(x_climate.shape)}")
        if x_climate.shape[1] != N_CLIMATE or x_climate.shape[2] != SEQ_LEN:
            raise ValueError(
                f"x_climate shape {tuple(x_climate.shape)} != ({N_CLIMATE}, {SEQ_LEN})"
            )
        if x_static.ndim != 2 or x_static.shape[1] != N_STATIC:
            raise ValueError(f"x_static must be [B, {N_STATIC}], got {tuple(x_static.shape)}")
        if x_climate.shape[0] != x_static.shape[0]:
            raise ValueError("batch size mismatch between climate and static inputs")

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        x_climate = batch["X_climate"]
        x_static = batch["X_static"]
        targets = torch.stack([batch["y_case2"], batch["y_almanac"]], dim=1)

        climate_grad = x_climate.detach().clone().requires_grad_(True)
        y_hat = self(climate_grad, x_static)
        mse = F.mse_loss(y_hat, targets)
        l_phys = physics_residual_loss(y_hat, climate_grad, lambda_phys=self.lambda_phys)
        loss = mse + l_phys

        self.log(f"{stage}_mse", mse, prog_bar=(stage == "train"))
        self.log(f"{stage}_phys", l_phys, prog_bar=False)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def predict_heads(self, x_climate: Tensor, x_static: Tensor) -> Tensor:
        """Deterministic forward (dropout off)."""
        was_training = self.training
        self.eval()
        out = self(x_climate, x_static)
        if was_training:
            self.train()
        return out

    def mc_predict_heads(
        self,
        x_climate: Tensor,
        x_static: Tensor,
        n_samples: int = 30,
    ) -> Tensor:
        """MC Dropout samples: ``[n_samples, batch, 2]`` in log1p space."""
        was_training = self.training
        self.eval()
        samples = torch.stack(
            [self(x_climate, x_static) for _ in range(n_samples)],
            dim=0,
        )
        if was_training:
            self.train()
        return samples


class YieldEnsemble:
    """
    Stacked ensemble over PINN CASE2/ALMANAC heads with deep + MC uncertainty.

    Parameters
    ----------
    models:
        One or more trained :class:`CocoaYieldPINN` checkpoints (multi-seed).
    n_mc_samples:
        MC Dropout passes per model at inference.
    """

    def __init__(
        self,
        models: list[CocoaYieldPINN] | None = None,
        n_mc_samples: int = 30,
    ) -> None:
        self.models: list[CocoaYieldPINN] = models or []
        self.n_mc_samples = n_mc_samples
        self._weights: dict[str, np.ndarray] = {}
        self._isotonic: dict[str, IsotonicRegression] = {}

    def add_model(self, model: CocoaYieldPINN) -> None:
        self.models.append(model)

    def fit_stacking(
        self,
        df: pd.DataFrame,
        ecozone_col: str = "ecozone",
        *,
        y_col: str = "y_true",
        case2_col: str = "pinn_case2",
        almanac_col: str = "pinn_almanac",
    ) -> None:
        """
        Per ecozone, fit NNLS stacking weights and isotonic calibration.

        Expects out-of-fold PINN predictions in ``df`` (columns ``pinn_case2``,
        ``pinn_almanac``) and ground-truth yields in ``y_col``, all in **log1p(kg/ha)** space.
        """
        required = {ecozone_col, y_col, case2_col, almanac_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"fit_stacking missing columns: {sorted(missing)}")

        self._weights.clear()
        self._isotonic.clear()

        for ecozone, group in df.groupby(ecozone_col):
            y = group[y_col].to_numpy(dtype=np.float64)
            a = group[case2_col].to_numpy(dtype=np.float64)
            b = group[almanac_col].to_numpy(dtype=np.float64)
            if len(y) < 2:
                logger.warning("ecozone %s has <2 rows; using equal weights", ecozone)
                w = np.array([0.5, 0.5], dtype=np.float64)
            else:
                w, _ = nnls(np.column_stack([a, b]), y)
                if w.sum() <= 0:
                    w = np.array([0.5, 0.5], dtype=np.float64)
                else:
                    w = w / w.sum()
            self._weights[str(ecozone)] = w

            raw = w[0] * a + w[1] * b
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw, y)
            self._isotonic[str(ecozone)] = iso

    def _stack_log1p(
        self,
        preds: np.ndarray,
        ecozones: np.ndarray,
    ) -> np.ndarray:
        """Apply ecozone weights to ``preds`` shaped ``[n, 2]`` (log1p)."""
        out = np.empty(preds.shape[0], dtype=np.float64)
        for i, eco in enumerate(ecozones):
            w = self._weights.get(str(eco), np.array([0.5, 0.5]))
            out[i] = w[0] * preds[i, 0] + w[1] * preds[i, 1]
        return out

    def _calibrate(self, stacked_log1p: np.ndarray, ecozones: np.ndarray) -> np.ndarray:
        out = stacked_log1p.copy()
        for eco, iso in self._isotonic.items():
            mask = ecozones == eco
            if mask.any():
                out[mask] = iso.predict(stacked_log1p[mask])
        return out

    def predict(
        self,
        X: dict[str, Tensor] | tuple[Tensor, Tensor],
        ecozones: np.ndarray | list[str] | None = None,
        *,
        return_uncertainty: bool = True,
    ) -> EnsemblePrediction | np.ndarray:
        """
        Deep ensemble (multi-seed) + MC Dropout uncertainty.

        Parameters
        ----------
        X:
            Dict with ``X_climate``, ``X_static`` or a tuple thereof.
        ecozones:
            Per-sample ecozone labels for stacking/calibration. Defaults to a
            single zone ``"default"`` using fitted weights or equal weights.
        """
        if not self.models:
            raise RuntimeError("YieldEnsemble has no PINN models; add_model() first.")

        if isinstance(X, dict):
            x_climate = X["X_climate"]
            x_static = X["X_static"]
        else:
            x_climate, x_static = X

        n = x_climate.shape[0]
        if ecozones is None:
            eco_arr = np.array(["default"] * n, dtype=object)
        else:
            eco_arr = np.asarray(ecozones, dtype=object)

        all_samples_kg: list[np.ndarray] = []
        for model in self.models:
            mc = model.mc_predict_heads(x_climate, x_static, n_samples=self.n_mc_samples)
            mc_log = mc.detach().cpu().numpy()  # [S, B, 2] log1p
            for s in range(mc_log.shape[0]):
                stacked_log = self._stack_log1p(mc_log[s], eco_arr)
                if self._isotonic:
                    stacked_log = self._calibrate(stacked_log, eco_arr)
                stacked_log = np.clip(stacked_log, -0.5, 12.0)
                all_samples_kg.append(np.expm1(stacked_log))

        samples = np.stack(all_samples_kg, axis=0)  # [n_total, B]
        mean = samples.mean(axis=0)
        if not return_uncertainty:
            return mean

        std = samples.std(axis=0)
        p10 = np.percentile(samples, 10, axis=0)
        p90 = np.percentile(samples, 90, axis=0)
        return EnsemblePrediction(mean=mean, std=std, p10=p10, p90=p90)

    def stacking_weights(self, ecozone: str) -> np.ndarray:
        """Return NNLS weights ``[w_case2, w_almanac]`` for an ecozone (sum to 1)."""
        return self._weights.get(ecozone, np.array([0.5, 0.5], dtype=np.float64))
