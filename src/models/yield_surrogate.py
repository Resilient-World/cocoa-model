"""
Neural surrogate for crop yield from daily climate time series and static soil features.

Designed to emulate slow process-based models (e.g. ALMANAC) with a fast forward pass.
This module is independent of the Sentinel/Prithvi segmentation stack; only the climate
tensor width (``climate_features``) must match your input variables.
"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class YieldPrediction(NamedTuple):
    """Monte Carlo yield estimate with epistemic uncertainty (std over forward passes)."""

    mean: Tensor
    std: Tensor


class MCDropout(nn.Module):
    """
    Dropout that stays active during inference for Monte Carlo uncertainty estimation.

    Unlike ``nn.Dropout``, forward always applies dropout (``training=True``),
    so repeated forward passes at inference time produce a predictive distribution.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, p=self.p, training=True)


class LossComponents(TypedDict):
    loss: Tensor
    mse: Tensor
    penalty: Tensor


class YieldSurrogateModel(nn.Module):
    """
    Two-branch yield predictor: LSTM over daily climate + MLP over static features.

    Parameters
    ----------
    sequence_length:
        Number of daily timesteps (default 365).
    climate_features:
        Variables per day, e.g. max temperature, min temperature, precipitation,
        radiation (default 4). To add variables (e.g. humidity), increase this
        and ensure your DataLoader provides ``[B, sequence_length, climate_features]``.
    static_features:
        Soil and management scalars per site (default 10).
    temporal_hidden:
        LSTM hidden size.
    static_hidden:
        Static MLP output width before fusion.
    head_hidden:
        Fusion MLP hidden width.
    lstm_layers:
        Stacked LSTM depth.
    dropout:
        Dropout probability on static branch, post-LSTM embedding, and fusion head.
        Uses :class:`MCDropout` so noise remains active at inference for MC sampling.
    """

    def __init__(
        self,
        sequence_length: int = 365,
        climate_features: int = 4,
        static_features: int = 10,
        temporal_hidden: int = 64,
        static_hidden: int = 64,
        head_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.climate_features = climate_features
        self.static_features = static_features

        self.climate_lstm = nn.LSTM(
            input_size=climate_features,
            hidden_size=temporal_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.climate_dropout = MCDropout(dropout)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_features, static_hidden),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(),
            MCDropout(dropout),
        )

        fusion_in = temporal_hidden + static_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_in, head_hidden),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def _validate_inputs(self, climate: Tensor, static: Tensor) -> None:
        if climate.ndim != 3:
            raise ValueError(
                f"climate must be [batch, sequence_length, climate_features], got shape {tuple(climate.shape)}"
            )
        if climate.shape[1] != self.sequence_length or climate.shape[2] != self.climate_features:
            raise ValueError(
                f"climate shape {tuple(climate.shape)} does not match "
                f"(sequence_length={self.sequence_length}, climate_features={self.climate_features})"
            )
        if static.ndim != 2 or static.shape[1] != self.static_features:
            raise ValueError(
                f"static must be [batch, static_features={self.static_features}], got {tuple(static.shape)}"
            )
        if climate.shape[0] != static.shape[0]:
            raise ValueError(
                f"batch size mismatch: climate {climate.shape[0]} vs static {static.shape[0]}"
            )

    def forward(self, climate: Tensor, static: Tensor) -> Tensor:
        """
        Parameters
        ----------
        climate:
            Daily climate tensor ``[batch, sequence_length, climate_features]``.
        static:
            Static features ``[batch, static_features]``.

        Returns
        -------
        Tensor
            Predicted yield ``[batch]`` (continuous scalar per sample).
        """
        self._validate_inputs(climate, static)

        _, (h_n, _) = self.climate_lstm(climate)
        climate_emb = self.climate_dropout(h_n[-1])

        static_emb = self.static_mlp(static)
        fused = torch.cat([climate_emb, static_emb], dim=1)
        return self.head(fused).squeeze(-1)


@torch.no_grad()
def predict_with_uncertainty(
    model: YieldSurrogateModel,
    x_climate: Tensor,
    x_static: Tensor,
    num_samples: int = 50,
) -> YieldPrediction:
    """
    Estimate yield and uncertainty via Monte Carlo Dropout.

    Runs ``num_samples`` stochastic forward passes (dropout active each time)
    and returns the mean prediction and standard deviation across samples.

    Parameters
    ----------
    model:
        Trained :class:`YieldSurrogateModel` with :class:`MCDropout` layers.
    x_climate:
        Daily climate tensor ``[batch, sequence_length, climate_features]``.
    x_static:
        Static features ``[batch, static_features]``.
    num_samples:
        Number of MC forward passes (default 50).

    Returns
    -------
    YieldPrediction
        ``mean``: final yield estimate ``[batch]``.
        ``std``: uncertainty (standard deviation across MC samples) ``[batch]``.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    was_training = model.training
    model.eval()

    samples = torch.stack(
        [model(x_climate, x_static) for _ in range(num_samples)],
        dim=0,
    )

    if was_training:
        model.train()

    # samples: [num_samples, batch]
    mean = samples.mean(dim=0)
    std = samples.std(dim=0) if num_samples > 1 else torch.zeros_like(mean)
    return YieldPrediction(mean=mean, std=std)


class PhysicsInformedYieldLoss(nn.Module):
    """
    MSE yield loss plus a penalty when predictions exceed a biophysical maximum.

    Encourages the ALMANAC surrogate to respect theoretical yield ceilings
    (e.g. crop-specific ``y_max`` from agronomic literature or calibration).
    """

    def __init__(
        self,
        y_max: float,
        penalty_weight: float = 100.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if y_max <= 0:
            raise ValueError(f"y_max must be positive, got {y_max}")
        if penalty_weight < 0:
            raise ValueError(f"penalty_weight must be non-negative, got {penalty_weight}")
        self.y_max = y_max
        self.penalty_weight = penalty_weight
        self.mse = nn.MSELoss(reduction=reduction)

    @staticmethod
    def _as_1d(tensor: Tensor) -> Tensor:
        if tensor.ndim > 1:
            return tensor.squeeze(-1)
        return tensor

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        *,
        return_components: bool = False,
    ) -> Tensor | LossComponents:
        pred = self._as_1d(pred)
        target = self._as_1d(target)

        mse = self.mse(pred, target)
        violation = F.relu(pred - self.y_max)
        penalty = self.penalty_weight * (violation**2).mean()
        total = mse + penalty

        if return_components:
            return LossComponents(
                loss=total,
                mse=mse.detach(),
                penalty=penalty.detach(),
            )
        return total
