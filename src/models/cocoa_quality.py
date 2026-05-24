"""Cocoa bean quality and premium-pricing feature model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

FERMENTATION_PRACTICES = ("traditional", "controlled", "none")
DRYING_METHODS = ("sun", "mechanical", "mixed")
QUALITY_INPUT_FEATURES = (
    "yield_t_per_ha",
    "harvest_window_precip_mm",
    "heat_stress_days_q3",
    "heat_stress_days_q4",
    "shade_cover_pct",
    "farm_age_years",
    "fermentation_traditional",
    "fermentation_controlled",
    "fermentation_none",
    "drying_sun",
    "drying_mechanical",
    "drying_mixed",
)


@dataclass(frozen=True)
class CocoaQualityInputs:
    yield_t_per_ha: float
    harvest_window_precip_mm: float
    heat_stress_days_q3: float
    heat_stress_days_q4: float
    shade_cover_pct: float
    fermentation_practice: str = "traditional"
    drying_method: str = "sun"
    farm_age_years: float = 12.0


@dataclass(frozen=True)
class CocoaQualityPrediction:
    fermentation_index: float
    defect_rate: float
    fine_flavor_probability: float
    price_premium_usd_per_t: float = 0.0


def encode_quality_inputs(inputs: CocoaQualityInputs) -> Tensor:
    """Pack scalar/categorical quality inputs into the model feature order."""
    if inputs.fermentation_practice not in FERMENTATION_PRACTICES:
        raise ValueError(f"Unknown fermentation practice: {inputs.fermentation_practice}")
    if inputs.drying_method not in DRYING_METHODS:
        raise ValueError(f"Unknown drying method: {inputs.drying_method}")

    values = [
        inputs.yield_t_per_ha / 4.0,
        inputs.harvest_window_precip_mm / 500.0,
        inputs.heat_stress_days_q3 / 90.0,
        inputs.heat_stress_days_q4 / 90.0,
        inputs.shade_cover_pct / 100.0,
        inputs.farm_age_years / 40.0,
    ]
    values.extend(float(inputs.fermentation_practice == name) for name in FERMENTATION_PRACTICES)
    values.extend(float(inputs.drying_method == name) for name in DRYING_METHODS)
    return torch.tensor(values, dtype=torch.float32)


class CocoaQualityModel(nn.Module):
    """
    Multi-output cocoa quality head.

    Outputs are ``fermentation_index`` (0–1), ``defect_rate`` (%), and
    ``fine_flavor_probability`` (0–1). The initialized baseline is monotonic
    for shade cover and controlled fermentation, so synthetic smoke tests are
    meaningful before ICCO/cooperative labels are integrated.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = len(QUALITY_INPUT_FEATURES)
        self.net = nn.Sequential(nn.Linear(self.input_dim, 3))
        self._init_baseline()

    def _init_baseline(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.net[-1]
        if not isinstance(final, nn.Linear):
            return
        with torch.no_grad():
            final.bias[:] = torch.tensor([0.0, -1.2, -0.8])
            final.weight[0, 1] = -0.8
            final.weight[0, 2] = -0.4
            final.weight[0, 3] = -0.3
            final.weight[0, 4] = 0.7
            final.weight[0, 7] = 1.1
            final.weight[0, 8] = -1.3
            final.weight[0, 9] = 0.3
            final.weight[0, 10] = 0.2

            final.weight[1, 1] = 1.5
            final.weight[1, 2] = 0.6
            final.weight[1, 3] = 0.5
            final.weight[1, 4] = -0.4
            final.weight[1, 7] = -0.9
            final.weight[1, 8] = 1.2
            final.weight[1, 10] = -0.4

            final.weight[2, 0] = 0.2
            final.weight[2, 1] = -0.7
            final.weight[2, 2] = -0.4
            final.weight[2, 3] = -0.3
            final.weight[2, 4] = 0.6
            final.weight[2, 7] = 1.0
            final.weight[2, 8] = -1.1
            final.weight[2, 9] = 0.2

    def forward(self, features: Tensor) -> Tensor:
        raw = self.net(features)
        fermentation = torch.sigmoid(raw[..., 0])
        defect_rate = 20.0 * torch.sigmoid(raw[..., 1])
        fine_flavor = torch.sigmoid(raw[..., 2])
        return torch.stack([fermentation, defect_rate, fine_flavor], dim=-1)

    @torch.no_grad()
    def predict_one(self, inputs: CocoaQualityInputs) -> CocoaQualityPrediction:
        features = encode_quality_inputs(inputs).unsqueeze(0)
        out = self(features).squeeze(0)
        return CocoaQualityPrediction(
            fermentation_index=float(out[0].item()),
            defect_rate=float(out[1].item()),
            fine_flavor_probability=float(out[2].item()),
        )


__all__ = [
    "DRYING_METHODS",
    "FERMENTATION_PRACTICES",
    "QUALITY_INPUT_FEATURES",
    "CocoaQualityInputs",
    "CocoaQualityModel",
    "CocoaQualityPrediction",
    "encode_quality_inputs",
]
