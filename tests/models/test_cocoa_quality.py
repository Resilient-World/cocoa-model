from __future__ import annotations

import torch

from models.cocoa_quality import CocoaQualityInputs, CocoaQualityModel, encode_quality_inputs


def test_cocoa_quality_shape_and_bounds() -> None:
    model = CocoaQualityModel()
    x = torch.stack(
        [
            encode_quality_inputs(
                CocoaQualityInputs(
                    yield_t_per_ha=1.5,
                    harvest_window_precip_mm=120.0,
                    heat_stress_days_q3=8.0,
                    heat_stress_days_q4=4.0,
                    shade_cover_pct=35.0,
                    fermentation_practice="traditional",
                    drying_method="sun",
                    farm_age_years=10.0,
                )
            ),
            encode_quality_inputs(
                CocoaQualityInputs(
                    yield_t_per_ha=2.1,
                    harvest_window_precip_mm=80.0,
                    heat_stress_days_q3=2.0,
                    heat_stress_days_q4=1.0,
                    shade_cover_pct=65.0,
                    fermentation_practice="controlled",
                    drying_method="mixed",
                    farm_age_years=12.0,
                )
            ),
        ]
    )
    y = model(x)
    assert y.shape == (2, 3)
    assert torch.all((y[:, 0] >= 0.0) & (y[:, 0] <= 1.0))
    assert torch.all((y[:, 1] >= 0.0) & (y[:, 1] <= 20.0))
    assert torch.all((y[:, 2] >= 0.0) & (y[:, 2] <= 1.0))


def test_controlled_fermentation_improves_quality_monotonicity() -> None:
    model = CocoaQualityModel()
    base = CocoaQualityInputs(
        yield_t_per_ha=1.5,
        harvest_window_precip_mm=120.0,
        heat_stress_days_q3=8.0,
        heat_stress_days_q4=4.0,
        shade_cover_pct=35.0,
        fermentation_practice="traditional",
        drying_method="sun",
        farm_age_years=10.0,
    )
    controlled = CocoaQualityInputs(
        yield_t_per_ha=1.5,
        harvest_window_precip_mm=120.0,
        heat_stress_days_q3=8.0,
        heat_stress_days_q4=4.0,
        shade_cover_pct=35.0,
        fermentation_practice="controlled",
        drying_method="sun",
        farm_age_years=10.0,
    )
    y_base = model(encode_quality_inputs(base).unsqueeze(0)).squeeze(0)
    y_controlled = model(encode_quality_inputs(controlled).unsqueeze(0)).squeeze(0)
    assert y_controlled[0] > y_base[0]
    assert y_controlled[1] < y_base[1]
    assert y_controlled[2] > y_base[2]
