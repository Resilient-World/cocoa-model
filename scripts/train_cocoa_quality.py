"""Train the cocoa quality model on synthetic quality labels.

TODO: replace/augment the synthetic fixture with ICCO and cooperative quality-lab
labels once data-sharing agreements are in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from models.cocoa_quality import CocoaQualityInputs, CocoaQualityModel, encode_quality_inputs


def _synthetic_batch(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    rows = []
    targets = []
    for i in range(n):
        yield_t = float(torch.rand((), generator=rng) * 3.0 + 0.4)
        precip = float(torch.rand((), generator=rng) * 450.0)
        heat_q3 = float(torch.rand((), generator=rng) * 45.0)
        heat_q4 = float(torch.rand((), generator=rng) * 35.0)
        shade = float(torch.rand((), generator=rng) * 80.0)
        controlled = i % 3 == 0
        none = i % 11 == 0
        drying = "mechanical" if precip > 260.0 else ("mixed" if precip > 160.0 else "sun")
        fermentation = "none" if none else ("controlled" if controlled else "traditional")
        inputs = CocoaQualityInputs(
            yield_t_per_ha=yield_t,
            harvest_window_precip_mm=precip,
            heat_stress_days_q3=heat_q3,
            heat_stress_days_q4=heat_q4,
            shade_cover_pct=shade,
            fermentation_practice=fermentation,
            drying_method=drying,
            farm_age_years=float(torch.rand((), generator=rng) * 30.0 + 2.0),
        )
        rows.append(encode_quality_inputs(inputs))
        fermentation_index = torch.sigmoid(
            torch.tensor(
                0.6
                + 0.012 * shade
                - 0.006 * precip
                + (0.8 if controlled else 0.0)
                - (1.4 if none else 0.0)
            )
        )
        defect_rate = torch.clamp(
            torch.tensor(
                2.0 + 0.025 * precip + 0.04 * heat_q4 - 0.035 * shade - (1.4 if controlled else 0.0)
            ),
            0.0,
            20.0,
        )
        fine_flavor = torch.sigmoid(
            torch.tensor(
                -1.2
                + 0.018 * shade
                - 0.004 * precip
                + (0.7 if controlled else 0.0)
                - 0.015 * heat_q3
            )
        )
        targets.append(torch.stack([fermentation_index, defect_rate, fine_flavor]))
    return torch.stack(rows), torch.stack(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("models/cocoa_quality.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x, y = _synthetic_batch(args.n, args.seed)
    model = CocoaQualityModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(args.epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "synthetic": True}, args.out)


if __name__ == "__main__":
    main()
