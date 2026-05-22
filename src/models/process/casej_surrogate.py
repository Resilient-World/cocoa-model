"""
Physics-informed CASEJ surrogate for CO2-aware scenario simulation.

Emulates the CASEJ cocoa process model (Asante et al. 2025) with explicit ``co2_ppm``
input and soft physics penalties for monotonic CO2 response, heat stress, water balance,
and shade-LAI VPD moderation.
"""

from __future__ import annotations

import structlog

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.surrogate.yield_surrogate import (
    CLIMATE_IDX,
    MCDropout,
    N_CLIMATE_CHANNELS,
    N_STATIC_SITE,
    YieldPrediction,
)

log = structlog.get_logger(__name__)

DEFAULT_CASEJ_CHECKPOINT = Path(__file__).resolve().parents[3] / "models" / "casej_surrogate.pt"
CASEJ_SLAI_STATIC_IDX = 5  # clay_frac slot reused for shade LAI norm in CASEJ training


class CASEJPhysicsLoss(nn.Module):
    """
    PINN penalties for :class:`CASEJSurrogate`.

    (a) Monotonic yield vs CO2 in 380–700 ppm
    (b) Heat-stress decline when cumulative degree-days above 32 °C increase
    (c) Water balance: yield ≤ f(annual ET, annual PPT)
    (d) Shade-LAI moderation of VPD stress
    """

    def __init__(
        self,
        lambda_mono: float = 0.1,
        lambda_heat: float = 0.1,
        lambda_water: float = 0.05,
        lambda_shade: float = 0.05,
        co2_low_ppm: float = 400.0,
        co2_high_ppm: float = 600.0,
        heat_threshold_c: float = 32.0,
    ) -> None:
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_heat = lambda_heat
        self.lambda_water = lambda_water
        self.lambda_shade = lambda_shade
        self.co2_low_ppm = co2_low_ppm
        self.co2_high_ppm = co2_high_ppm
        self.heat_threshold_c = heat_threshold_c
        self.mse = nn.MSELoss()

    @staticmethod
    def _annual_precip_et(climate: Tensor) -> tuple[Tensor, Tensor]:
        precip = climate[..., CLIMATE_IDX["precip"]].sum(dim=1)
        et0 = climate[..., CLIMATE_IDX["et0"]].sum(dim=1)
        return precip, et0

    @staticmethod
    def _heat_cdd(climate: Tensor) -> Tensor:
        tmax = climate[..., CLIMATE_IDX["tmax"]]
        return F.relu(tmax - 32.0).sum(dim=1)

    def monotonicity_penalty(
        self,
        model: CASEJSurrogate,
        climate: Tensor,
        static: Tensor,
    ) -> Tensor:
        co2_low = torch.full((climate.size(0),), self.co2_low_ppm, device=climate.device, dtype=climate.dtype)
        co2_high = torch.full((climate.size(0),), self.co2_high_ppm, device=climate.device, dtype=climate.dtype)
        y_low = model(climate, static, co2_ppm=co2_low)
        y_high = model(climate, static, co2_ppm=co2_high)
        return F.relu(y_low - y_high).mean()

    def heat_penalty(self, pred: Tensor, climate: Tensor) -> Tensor:
        cdd = self._heat_cdd(climate)
        excess_cdd = F.relu(cdd - 30.0)
        return (F.relu(pred - 0.5) * excess_cdd).mean()

    def water_penalty(self, pred: Tensor, climate: Tensor) -> Tensor:
        ppt, et = self._annual_precip_et(climate)
        cap = 0.35 + 1.8 * (ppt / et.clamp(min=1.0))
        cap = cap.clamp(0.2, 3.5)
        return F.relu(pred - cap).pow(2).mean()

    def shade_vpd_penalty(self, pred: Tensor, climate: Tensor, static: Tensor) -> Tensor:
        if static.shape[1] <= CASEJ_SLAI_STATIC_IDX:
            return pred.new_tensor(0.0)
        slai = static[:, CASEJ_SLAI_STATIC_IDX].clamp(0.0, 1.0)
        vpd_mean = climate[..., CLIMATE_IDX["vpd"]].mean(dim=1)
        high_vpd = F.relu(vpd_mean - 1.5)
        unshaded = 1.0 - slai
        return (F.relu(pred - 1.0) * high_vpd * unshaded).mean()

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        model: CASEJSurrogate,
        climate: Tensor,
        static: Tensor,
        *,
        return_components: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        target = target.squeeze(-1) if target.ndim > 1 else target

        mse = self.mse(pred, target)
        p_mono = self.monotonicity_penalty(model, climate, static)
        p_heat = self.heat_penalty(pred, climate)
        p_water = self.water_penalty(pred, climate)
        p_shade = self.shade_vpd_penalty(pred, climate, static)

        total = (
            mse
            + self.lambda_mono * p_mono
            + self.lambda_heat * p_heat
            + self.lambda_water * p_water
            + self.lambda_shade * p_shade
        )
        if return_components:
            return {
                "loss": total,
                "mse": mse.detach(),
                "penalty_mono": p_mono.detach(),
                "penalty_heat": p_heat.detach(),
                "penalty_water": p_water.detach(),
                "penalty_shade": p_shade.detach(),
            }
        return total


class CASEJSurrogate(nn.Module):
    """
    Dual-branch PINN: LSTM climate encoder + MLP static + explicit CO2 embedding.

    ``forward(climate, static, co2_ppm=...)`` — ``co2_ppm`` overrides the climate
    channel when provided (required for valid SSP scenario extrapolation).
    """

    def __init__(
        self,
        sequence_length: int = 365,
        climate_features: int = N_CLIMATE_CHANNELS,
        static_features: int = N_STATIC_SITE,
        lstm_hidden: int = 96,
        lstm_layers: int = 2,
        static_hidden: int = 64,
        co2_embed_dim: int = 16,
        head_hidden: int = 64,
        dropout: float = 0.1,
        galileo_dim: int = 0,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.climate_features = climate_features
        self.site_static_features = static_features
        self.static_features = static_features + galileo_dim
        self.galileo_dim = galileo_dim

        self.climate_lstm = nn.LSTM(
            input_size=climate_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * 2
        self.climate_dropout = MCDropout(dropout)

        self.co2_mlp = nn.Sequential(
            nn.Linear(1, co2_embed_dim),
            nn.ReLU(),
            MCDropout(dropout),
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(self.static_features, static_hidden),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(),
            MCDropout(dropout),
        )
        fusion = lstm_out + static_hidden + co2_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fusion, head_hidden),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self._yield_floor = 0.05

    def _inject_co2(self, climate: Tensor, co2_ppm: Tensor | None) -> Tensor:
        if co2_ppm is None:
            return climate
        out = climate.clone()
        ppm = co2_ppm.view(-1, 1, 1).expand(-1, climate.size(1), 1)
        out[..., CLIMATE_IDX["co2_ppm"]] = ppm.squeeze(-1)
        return out

    def forward(
        self,
        climate: Tensor,
        static: Tensor,
        co2_ppm: Tensor | None = None,
    ) -> Tensor:
        if climate.ndim != 3 or static.ndim != 2:
            raise ValueError(
                f"Expected climate [B,T,C] and static [B,F]; got {tuple(climate.shape)}, {tuple(static.shape)}"
            )
        climate = self._inject_co2(climate, co2_ppm)
        lstm_out, _ = self.climate_lstm(climate)
        climate_emb = self.climate_dropout(lstm_out[:, -1, :])
        static_emb = self.static_mlp(static)
        if co2_ppm is None:
            co2_scalar = climate[:, 0, CLIMATE_IDX["co2_ppm"]].mean(dim=1, keepdim=True)
        else:
            co2_scalar = co2_ppm.view(-1, 1).to(climate.dtype)
        co2_emb = self.co2_mlp(co2_scalar)
        fused = torch.cat([climate_emb, static_emb, co2_emb], dim=1)
        raw = self.head(fused).squeeze(-1)
        return F.softplus(raw) + self._yield_floor


@torch.no_grad()
def predict_casej_with_uncertainty(
    model: CASEJSurrogate,
    climate: Tensor,
    static: Tensor,
    co2_ppm: Tensor,
    num_samples: int = 50,
) -> YieldPrediction:
    was_training = model.training
    model.eval()
    samples = torch.stack(
        [
            model(climate, static, co2_ppm=co2_ppm).squeeze(0)
            for _ in range(num_samples)
        ],
        dim=0,
    )
    if was_training:
        model.train()
    mean = samples.mean(dim=0)
    std = samples.std(dim=0) if num_samples > 1 else torch.zeros_like(mean)
    return YieldPrediction(mean=mean, std=std)


def load_casej_surrogate(
    checkpoint: Path | str | None = None,
    *,
    galileo_dim: int = 0,
    device: str = "cpu",
) -> CASEJSurrogate:
    path = Path(checkpoint) if checkpoint else DEFAULT_CASEJ_CHECKPOINT
    model = CASEJSurrogate(galileo_dim=galileo_dim).to(device)
    if path.is_file():
        state = torch.load(path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        log.info("Loaded CASEJSurrogate from %s", path)
    else:
        log.warning("CASEJ checkpoint missing at %s; using uninitialized weights", path)
    model.eval()
    return model
