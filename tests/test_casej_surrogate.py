"""Physics and CO2 constraints for CASEJ process model and PINN surrogate."""

from __future__ import annotations

import numpy as np
import torch

from models.casej_process import (
    CASEJSite,
    co2_fertilization_factor,
    load_casej_params,
    run_casej_yearly,
    synthesize_daily_weather,
)
from models.casej_surrogate import CASEJPhysicsLoss, CASEJSurrogate


def test_casej_process_co2_monotonic() -> None:
    params = load_casej_params()
    weather = synthesize_daily_weather(365, seed=1)
    site_low = CASEJSite(6.0, -3.0, 150.0, 1.0, 12.0, 400.0)
    site_high = CASEJSite(6.0, -3.0, 150.0, 1.0, 12.0, 600.0)
    y_low = run_casej_yearly(weather, site_low, params)["yield_t_ha"]
    y_high = run_casej_yearly(weather, site_high, params)["yield_t_ha"]
    assert y_high > y_low


def test_casej_process_heat_decline() -> None:
    params = load_casej_params()
    weather = synthesize_daily_weather(365, seed=2)
    site = CASEJSite(6.0, -3.0, 150.0, 1.0, 12.0, 420.0)
    y_base = run_casej_yearly(weather, site, params)["yield_t_ha"]

    hot = weather.copy()
    hot["tmax_c"] = hot["tmax_c"] + 4.0
    hot["tmean_c"] = 0.5 * (hot["tmax_c"] + hot["tmin_c"])
    y_hot = run_casej_yearly(hot, site, params)["yield_t_ha"]
    assert y_hot < y_base


def test_co2_factor_monotonic_in_range() -> None:
    params = load_casej_params()
    ppm = np.linspace(380, 700, 12)
    f = co2_fertilization_factor(ppm, params)
    assert np.all(np.diff(f) >= -1e-9)


def test_casej_surrogate_co2_monotonic_after_training() -> None:
    """Short PINN fit on synthetic monotonic CO2 labels."""
    torch.manual_seed(0)
    model = CASEJSurrogate()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = CASEJPhysicsLoss(lambda_mono=0.2)

    climate = torch.randn(32, 365, 11)
    static = torch.randn(32, 13)
    co2 = torch.linspace(400, 600, 32)
    target = 0.5 + 0.003 * (co2 - 400)

    for _ in range(80):
        opt.zero_grad()
        pred = model(climate, static, co2_ppm=co2)
        loss = loss_fn(pred, target, model, climate, static)
        loss.backward()
        opt.step()

    model.eval()
    climate_one = climate[:1]
    static_one = static[:1]
    co2_grid = torch.tensor([400.0, 450.0, 500.0, 550.0, 600.0])
    yields = [model(climate_one, static_one, co2_ppm=c.view(1)).item() for c in co2_grid]
    assert all(yields[i + 1] >= yields[i] - 1e-4 for i in range(len(yields) - 1))


def test_casej_surrogate_heat_penalty_positive_on_bad_pred() -> None:
    climate = torch.zeros(4, 365, 11)
    climate[..., 0] = 38.0  # tmax channel
    static = torch.zeros(4, 13)
    model = CASEJSurrogate()
    pred = torch.ones(4) * 2.5
    loss_fn = CASEJPhysicsLoss()
    heat_p = loss_fn.heat_penalty(pred, climate)
    assert float(heat_p) > 0.0


def test_physics_loss_monotonicity_penalizes_violation() -> None:
    model = CASEJSurrogate()

    class _StubModel(torch.nn.Module):
        def forward(self, climate, static, co2_ppm=None):
            if co2_ppm is not None and co2_ppm[0] < 500:
                return torch.tensor([1.0])
            return torch.tensor([0.5])

    climate = torch.zeros(1, 365, 11)
    static = torch.zeros(1, 13)
    stub = _StubModel()
    loss_fn = CASEJPhysicsLoss()
    p = loss_fn.monotonicity_penalty(stub, climate, static)  # type: ignore[arg-type]
    assert float(p) > 0.0
