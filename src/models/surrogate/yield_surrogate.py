"""
Physics-informed neural surrogate for cocoa yield from daily climate and static site features.

Combines a daily mechanistic core (GDD, soil water, VPD/temperature/CO₂ stress, RUE biomass)
with a bidirectional GRU + attention residual corrector. Designed as a fast stand-in for
process-based models (e.g. ALMANAC) while keeping ``forward(climate, static) -> [B]`` stable
for the intervention API.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Channel / feature contracts
# ---------------------------------------------------------------------------

CLIMATE_CHANNEL_NAMES: tuple[str, ...] = (
    "tmax",
    "tmin",
    "tmean",
    "precip",
    "srad",
    "vpd",
    "et0",
    "sm_root",
    "wind10m",
    "rh_mean",
    "co2_ppm",
)

N_CLIMATE_CHANNELS = len(CLIMATE_CHANNEL_NAMES)
CLIMATE_IDX = {name: i for i, name in enumerate(CLIMATE_CHANNEL_NAMES)}

# Static feature registry (default 13 site features; index 0 = AWC mm).
STATIC_FEATURE_NAMES: tuple[str, ...] = (
    "awc_mm",
    "sand_frac",
    "baseline_yield_scaled",
    "intervention_flag",
    "stress_tolerance_flag",
    "clay_frac",
    "soc_norm",
    "ph_norm",
    "treecover_norm",
    "cocoa_prob",
    "tree_age_years_norm",
    "cohort_phase",
    "planting_density_norm",
)

N_STATIC_SITE = len(STATIC_FEATURE_NAMES)
STATIC_IDX = {name: i for i, name in enumerate(STATIC_FEATURE_NAMES)}

# Phenology cohort multipliers (applied as mechanistic f_age).
COHORT_JUVENILE = 0.0  # < 5 y
COHORT_RAMP = 0.5  # 5–10 y
COHORT_PEAK = 1.0  # 10–25 y
COHORT_SENESCENT = 0.6  # > 25 y

DEFAULT_TREE_AGE_YEARS = 12.0
DEFAULT_PLANTING_DENSITY = 1100.0
MAX_TREE_AGE_NORM_YEARS = 40.0
MAX_PLANTING_DENSITY = 1500.0


def cohort_phase_from_age(age_years: float) -> float:
    """Map tree age (years) to cohort phase multiplier for mechanistic f_age."""
    age = float(age_years)
    if age < 5.0:
        return COHORT_JUVENILE
    if age < 10.0:
        return COHORT_RAMP
    if age <= 25.0:
        return COHORT_PEAK
    return COHORT_SENESCENT


def tree_age_years_norm(age_years: float, *, max_age: float = MAX_TREE_AGE_NORM_YEARS) -> float:
    return float(min(max(float(age_years) / max_age, 0.0), 1.0))


def planting_density_norm(
    density_trees_ha: float,
    *,
    max_density: float = MAX_PLANTING_DENSITY,
) -> float:
    return float(min(max(float(density_trees_ha) / max_density, 0.0), 1.0))


def pack_tree_age_static(
    age_years: float,
    *,
    planting_density_trees_ha: float = DEFAULT_PLANTING_DENSITY,
) -> tuple[float, float, float]:
    """Return ``(tree_age_years_norm, cohort_phase, planting_density_norm)``."""
    return (
        tree_age_years_norm(age_years),
        cohort_phase_from_age(age_years),
        planting_density_norm(planting_density_trees_ha),
    )

# Legacy 4-channel order (geo_mock / early API): tmax, tmin, precip, srad
_LEGACY_4_NAMES: tuple[str, ...] = ("tmax", "tmin", "precip", "srad")

# Mechanistic constants (Lahive 2019, FAO-56 cocoa, Bastide 2009, Long et al. 2004)
GDD_VEG_BASE = 18.7
GDD_POD_BASE = 9.0
GDD_CAP = 32.0
VPD_BREAKPOINT_KPA = 1.65
VPD_SLOPE = 0.2
VPD_PENALTY_KPA = 1.8
TEMP_OPT_C = 32.0
TEMP_MIN_C = 18.0
TEMP_MAX_C = 40.0
CO2_REF_PPM = 400.0
CO2_GAIN = 0.38
CO2_SCALE_PPM = 300.0
CO2_F_MAX = 1.4
KC_COCOA = 1.05
AWC_STATIC_IDX = 0
GRAMS_PER_TONNE = 1_000_000.0


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


class MechanisticTraces(TypedDict):
    y_mech: Tensor
    stress_trace: Tensor
    biomass_trace: Tensor
    sw_trace: Tensor


class AttentionPool(nn.Module):
    """Single learned query attending over the temporal GRU sequence (multi-head)."""

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pool ``[B, T, D]`` → ``[B, D]``."""
        batch = x.size(0)
        q = self.query.expand(batch, -1, -1)
        pooled, _ = self.attn(q, x, x, need_weights=False)
        return pooled.squeeze(1)


class MechanisticCore(nn.Module):
    """
    Daily process-based cocoa growth loop (differentiable, Python time-step).

    Returns mechanistic yield (t/ha) and diagnostic traces.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rue = nn.Parameter(torch.tensor(1.8))
        self.harvest_index = nn.Parameter(torch.tensor(0.12))

    @staticmethod
    def _temp_stress(tmean: Tensor) -> Tensor:
        """Piecewise-linear: 0 at 18 °C and 40 °C, peak 1 at 32 °C."""
        rise = (tmean - TEMP_MIN_C) / (TEMP_OPT_C - TEMP_MIN_C)
        fall = (TEMP_MAX_C - tmean) / (TEMP_MAX_C - TEMP_OPT_C)
        return torch.clamp(torch.minimum(rise, fall), 0.0, 1.0)

    @staticmethod
    def _vpd_stress(vpd: Tensor) -> Tensor:
        return 1.0 - torch.sigmoid((vpd - VPD_BREAKPOINT_KPA) / VPD_SLOPE)

    @staticmethod
    def _co2_factor(co2: Tensor) -> Tensor:
        raw = 1.0 + CO2_GAIN * (co2 - CO2_REF_PPM) / CO2_SCALE_PPM
        return torch.clamp(raw, max=CO2_F_MAX)

    def forward(
        self,
        climate: Tensor,
        static: Tensor,
    ) -> MechanisticTraces:
        """
        Parameters
        ----------
        climate:
            ``[B, T, 11]`` — channels per :data:`CLIMATE_CHANNEL_NAMES`.
        static:
            ``[B, F]`` — index 0 = available water capacity (mm).
        """
        batch, time_steps, _ = climate.shape
        device = climate.device
        dtype = climate.dtype

        tmean = climate[..., CLIMATE_IDX["tmean"]]
        precip = climate[..., CLIMATE_IDX["precip"]].clamp(min=0.0)
        srad = climate[..., CLIMATE_IDX["srad"]].clamp(min=0.0)
        vpd = climate[..., CLIMATE_IDX["vpd"]]
        et0 = climate[..., CLIMATE_IDX["et0"]].clamp(min=0.0)
        co2 = climate[..., CLIMATE_IDX["co2_ppm"]]

        awc = static[:, AWC_STATIC_IDX].clamp(min=1.0).unsqueeze(1)

        gdd_veg_daily = (tmean - GDD_VEG_BASE).clamp(min=0.0, max=GDD_CAP - GDD_VEG_BASE)
        gdd_pod_daily = (tmean - GDD_POD_BASE).clamp(min=0.0, max=GDD_CAP - GDD_POD_BASE)
        _ = gdd_veg_daily.cumsum(dim=1)
        _ = gdd_pod_daily.cumsum(dim=1)

        sw = torch.zeros(batch, device=device, dtype=dtype)
        sw_trace = torch.zeros(batch, time_steps, device=device, dtype=dtype)
        biomass_trace = torch.zeros(batch, time_steps, device=device, dtype=dtype)
        stress_trace = torch.zeros(batch, time_steps, 5, device=device, dtype=dtype)

        cohort_idx = STATIC_IDX["cohort_phase"]
        if static.shape[1] > cohort_idx:
            f_age = static[:, cohort_idx].clamp(0.0, 1.0)
        else:
            f_age = torch.full(
                (batch,),
                cohort_phase_from_age(DEFAULT_TREE_AGE_YEARS),
                device=device,
                dtype=dtype,
            )

        rue = self.rue.clamp(min=0.1)
        hi = self.harvest_index.clamp(0.01, 0.5)
        biomass_cum = torch.zeros(batch, device=device, dtype=dtype)

        for t in range(time_steps):
            sw_trace[:, t] = sw
            f_w = torch.clamp(sw / (0.5 * awc.squeeze(1)), 0.0, 1.0)
            f_vpd = self._vpd_stress(vpd[:, t])
            f_temp = self._temp_stress(tmean[:, t])
            f_co2 = self._co2_factor(co2[:, t])

            stress_trace[:, t, 0] = f_w
            stress_trace[:, t, 1] = f_vpd
            stress_trace[:, t, 2] = f_temp
            stress_trace[:, t, 3] = f_co2
            stress_trace[:, t, 4] = f_age

            d_b = rue * srad[:, t] * f_w * f_vpd * f_temp * f_co2 * f_age
            biomass_cum = biomass_cum + d_b
            biomass_trace[:, t] = biomass_cum

            et_crop = et0[:, t] * KC_COCOA
            sw = (sw + precip[:, t] - et_crop).clamp(min=0.0).clamp(max=awc.squeeze(1))

        y_mech = hi * biomass_cum / GRAMS_PER_TONNE

        return MechanisticTraces(
            y_mech=y_mech,
            stress_trace=stress_trace,
            biomass_trace=biomass_trace,
            sw_trace=sw_trace,
        )


class YieldSurrogateModel(nn.Module):
    """
    PINN yield model: mechanistic core + neural residual on climate/static embeddings.

    Parameters
    ----------
    sequence_length:
        Daily timesteps (default 365).
    climate_features:
        Input channels per day. Default 11 (:data:`CLIMATE_CHANNEL_NAMES`). Legacy ``4``
        pads missing channels with zeros (deprecation warning).
    static_features:
        Site covariates (default 13); index 0 = AWC (mm) for the mechanistic soil bucket.
    galileo_dim:
        Optional Galileo embedding width appended to site static features
        (total input width = ``static_features + galileo_dim``).
    static_hidden, head_hidden, dropout:
        MLP / residual head hyperparameters.
    gru_hidden, gru_layers, attn_heads:
        Temporal encoder settings (bidirectional GRU + attention pool).
    """

    climate_channel_names: tuple[str, ...] = CLIMATE_CHANNEL_NAMES

    def __init__(
        self,
        sequence_length: int = 365,
        climate_features: int = N_CLIMATE_CHANNELS,
        static_features: int = N_STATIC_SITE,
        static_feature_names: tuple[str, ...] | None = None,
        galileo_dim: int = 0,
        static_hidden: int = 64,
        head_hidden: int = 64,
        gru_hidden: int = 96,
        gru_layers: int = 2,
        attn_heads: int = 4,
        dropout: float = 0.1,
        # Legacy kwargs (ignored, kept for checkpoint / test compatibility)
        temporal_hidden: int | None = None,
        lstm_layers: int | None = None,
    ) -> None:
        super().__init__()
        _ = temporal_hidden, lstm_layers

        self.sequence_length = sequence_length
        self._legacy_input_width = climate_features == 4
        self.climate_features = (
            N_CLIMATE_CHANNELS if self._legacy_input_width else climate_features
        )
        if galileo_dim < 0:
            raise ValueError(f"galileo_dim must be >= 0, got {galileo_dim}")
        self.galileo_dim = galileo_dim
        if static_feature_names is not None:
            static_feature_names = tuple(static_feature_names)
            static_features = len(static_feature_names)
        self.static_feature_names = static_feature_names or STATIC_FEATURE_NAMES
        self.site_static_features = static_features
        total_static = static_features + galileo_dim
        self.static_features = total_static

        self.mechanistic = MechanisticCore()

        gru_in = self.climate_features
        self.climate_gru = nn.GRU(
            input_size=gru_in,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * 2
        self.attention_pool = AttentionPool(gru_out_dim, num_heads=attn_heads)
        self.climate_dropout = MCDropout(dropout)

        self.static_mlp = nn.Sequential(
            nn.Linear(total_static, static_hidden),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(),
            MCDropout(dropout),
        )

        stress_summary_dim = 5
        fusion_in = gru_out_dim + static_hidden + stress_summary_dim
        self.residual_head = nn.Sequential(
            nn.Linear(fusion_in, head_hidden),
            nn.ReLU(),
            MCDropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self._init_small_residual()

    def _init_small_residual(self) -> None:
        last = self.residual_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, std=0.01)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def _pad_climate(self, climate: Tensor) -> Tensor:
        """Expand legacy 4-channel inputs to full 11-channel tensor."""
        width = climate.shape[2]
        if width == self.climate_features:
            return climate
        if width == 4 and self._legacy_input_width:
            warnings.warn(
                "climate_features=4 is deprecated; padding to 11 channels "
                f"({', '.join(CLIMATE_CHANNEL_NAMES)}). "
                "Pass full ERA5 channel order or set climate_features=11.",
                DeprecationWarning,
                stacklevel=3,
            )
            padded = climate.new_zeros(climate.shape[0], climate.shape[1], N_CLIMATE_CHANNELS)
            for i, name in enumerate(_LEGACY_4_NAMES):
                padded[..., CLIMATE_IDX[name]] = climate[..., i]
            return padded
        if width == 4 and not self._legacy_input_width:
            warnings.warn(
                "Received 4 climate channels on an 11-channel model; deprecated — "
                "zero-padding extras to 11 channels.",
                DeprecationWarning,
                stacklevel=3,
            )
            padded = climate.new_zeros(climate.shape[0], climate.shape[1], N_CLIMATE_CHANNELS)
            for i, name in enumerate(_LEGACY_4_NAMES):
                padded[..., CLIMATE_IDX[name]] = climate[..., i]
            return padded
        raise ValueError(
            f"climate width {width} incompatible with configured "
            f"climate_features={self.climate_features}"
        )

    def _validate_inputs(self, climate: Tensor, static: Tensor) -> None:
        if climate.ndim != 3:
            raise ValueError(
                f"climate must be [batch, sequence_length, climate_features], "
                f"got shape {tuple(climate.shape)}"
            )
        expected_w = 4 if self._legacy_input_width else self.climate_features
        if climate.shape[1] != self.sequence_length:
            raise ValueError(
                f"climate shape {tuple(climate.shape)} does not match "
                f"(sequence_length={self.sequence_length}, climate_features={expected_w})"
            )
        if climate.shape[2] not in (4, expected_w, N_CLIMATE_CHANNELS):
            raise ValueError(
                f"climate shape {tuple(climate.shape)} does not match "
                f"(sequence_length={self.sequence_length}, climate_features={expected_w})"
            )
        if static.ndim != 2 or static.shape[1] != self.static_features:
            raise ValueError(
                f"static must be [batch, static_features={self.static_features}], "
                f"got {tuple(static.shape)}"
            )
        if climate.shape[0] != static.shape[0]:
            raise ValueError(
                f"batch size mismatch: climate {climate.shape[0]} vs static {static.shape[0]}"
            )

    def _encode(
        self,
        climate: Tensor,
        static: Tensor,
        traces: MechanisticTraces,
    ) -> Tensor:
        """Neural residual correction (t/ha), initialized small."""
        gru_out, _ = self.climate_gru(climate)
        climate_emb = self.climate_dropout(self.attention_pool(gru_out))
        static_emb = self.static_mlp(static)
        stress_summary = traces["stress_trace"].mean(dim=1)
        fused = torch.cat([climate_emb, static_emb, stress_summary], dim=1)
        return self.residual_head(fused).squeeze(-1)

    def forward_with_traces(
        self,
        climate: Tensor,
        static: Tensor,
    ) -> tuple[Tensor, MechanisticTraces]:
        self._validate_inputs(climate, static)
        climate_full = self._pad_climate(climate)
        traces = self.mechanistic(climate_full, static)
        residual = self._encode(climate_full, static, traces)
        y = traces["y_mech"] + residual
        return y, traces

    def forward_with_activations(
        self,
        climate: Tensor,
        static: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(yield, hidden)`` where hidden is the fusion layer before the residual head."""
        self._validate_inputs(climate, static)
        climate_full = self._pad_climate(climate)
        traces = self.mechanistic(climate_full, static)
        gru_out, _ = self.climate_gru(climate_full)
        climate_emb = self.climate_dropout(self.attention_pool(gru_out))
        static_emb = self.static_mlp(static)
        stress_summary = traces["stress_trace"].mean(dim=1)
        hidden = torch.cat([climate_emb, static_emb, stress_summary], dim=1)
        residual = self.residual_head(hidden).squeeze(-1)
        y = traces["y_mech"] + residual
        return y, hidden

    def forward(self, climate: Tensor, static: Tensor) -> Tensor:
        y, _ = self.forward_with_traces(climate, static)
        return y


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

    mean = samples.mean(dim=0)
    std = samples.std(dim=0) if num_samples > 1 else torch.zeros_like(mean)
    return YieldPrediction(mean=mean, std=std)


class CocoaPINNLoss(nn.Module):
    """
    Multi-term PINN loss for :class:`YieldSurrogateModel`.

    Combines MSE, yield ceiling, VPD / soil-water / biomass-monotonicity constraints,
    and optional harvest-day auxiliary loss.
    """

    def __init__(
        self,
        y_max: float = 3.5,
        lambda_max: float = 10.0,
        lambda_vpd: float = 0.5,
        lambda_water: float = 100.0,
        lambda_mono: float = 1.0,
        lambda_aux: float = 0.5,
        vpd_penalty_kpa: float = VPD_PENALTY_KPA,
    ) -> None:
        super().__init__()
        if y_max <= 0:
            raise ValueError(f"y_max must be positive, got {y_max}")
        self.y_max = y_max
        self.lambda_max = lambda_max
        self.lambda_vpd = lambda_vpd
        self.lambda_water = lambda_water
        self.lambda_mono = lambda_mono
        self.lambda_aux = lambda_aux
        self.vpd_penalty_kpa = vpd_penalty_kpa
        self.mse = nn.MSELoss()

    @staticmethod
    def _as_1d(tensor: Tensor) -> Tensor:
        if tensor.ndim > 1:
            return tensor.squeeze(-1)
        return tensor

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        traces: MechanisticTraces,
        climate: Tensor | None = None,
        *,
        predicted_harvest_day: Tensor | None = None,
        observed_harvest_day: Tensor | None = None,
        return_components: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        pred = self._as_1d(pred)
        target = self._as_1d(target)

        mse = self.mse(pred, target)
        penalty_max = self.lambda_max * (F.relu(pred - self.y_max) ** 2).mean()

        vpd = traces["stress_trace"][..., 1]
        if climate is not None:
            climate_full = climate
            if climate.shape[-1] < N_CLIMATE_CHANNELS:
                pad = climate.new_zeros(*climate.shape[:-1], N_CLIMATE_CHANNELS)
                for i, name in enumerate(_LEGACY_4_NAMES):
                    if i < climate.shape[-1]:
                        pad[..., CLIMATE_IDX[name]] = climate[..., i]
                climate_full = pad
            vpd_raw = climate_full[..., CLIMATE_IDX["vpd"]]
            vpd_excess = F.relu(vpd_raw - self.vpd_penalty_kpa)
            penalty_vpd = self.lambda_vpd * (vpd_excess * F.relu(vpd)).mean()
        else:
            penalty_vpd = self.lambda_vpd * F.relu(vpd).mean()

        penalty_water = self.lambda_water * F.relu(-traces["sw_trace"]).mean()

        biomass = traces["biomass_trace"]
        d_biomass = biomass[:, 1:] - biomass[:, :-1]
        penalty_mono = self.lambda_mono * F.relu(-d_biomass).mean()

        total = mse + penalty_max + penalty_vpd + penalty_water + penalty_mono

        penalty_aux = pred.new_tensor(0.0)
        if predicted_harvest_day is not None and observed_harvest_day is not None:
            penalty_aux = self.lambda_aux * self.mse(
                self._as_1d(predicted_harvest_day),
                self._as_1d(observed_harvest_day),
            )
            total = total + penalty_aux

        if return_components:
            return {
                "loss": total,
                "mse": mse.detach(),
                "penalty_max": penalty_max.detach(),
                "penalty_vpd": penalty_vpd.detach(),
                "penalty_water": penalty_water.detach(),
                "penalty_mono": penalty_mono.detach(),
                "penalty_aux": penalty_aux.detach(),
            }
        return total


class PhysicsInformedYieldLoss(nn.Module):
    """
    MSE yield loss plus a penalty when predictions exceed a biophysical maximum.

    Retained for backward compatibility with early tests and training scripts.
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


class DeepEnsemblePrediction(NamedTuple):
    mean: Tensor
    std: Tensor


class DeepEnsemble(nn.Module):
    """
    Deep ensemble of :class:`YieldSurrogateModel` members with MC-dropout uncertainty.

    Total variance combines across-member spread and within-member MC variance.
    """

    def __init__(
        self,
        n_members: int = 5,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        kwargs = model_kwargs or {}
        self.members = nn.ModuleList(
            [YieldSurrogateModel(**kwargs) for _ in range(n_members)]
        )

    def predict(
        self,
        climate: Tensor,
        static: Tensor,
        *,
        num_mc_samples: int = 10,
    ) -> DeepEnsemblePrediction:
        """
        Returns ensemble mean/std over members, aggregating MC dropout variance.
        """
        member_means: list[Tensor] = []
        member_vars: list[Tensor] = []

        for member in self.members:
            mc = predict_with_uncertainty(
                member,
                climate,
                static,
                num_samples=num_mc_samples,
            )
            member_means.append(mc.mean)
            member_vars.append(mc.std**2)

        means = torch.stack(member_means, dim=0)
        vars_ = torch.stack(member_vars, dim=0)

        ensemble_mean = means.mean(dim=0)
        # Var(total) = E[Var(Y|M)] + Var(E[Y|M])
        total_var = vars_.mean(dim=0) + means.var(dim=0, unbiased=False)
        total_std = torch.sqrt(total_var.clamp(min=0.0))
        return DeepEnsemblePrediction(mean=ensemble_mean, std=total_std)
