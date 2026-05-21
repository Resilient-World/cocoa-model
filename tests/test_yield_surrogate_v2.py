"""Tests for YieldSurrogateV2 and v1 checkpoint migration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.yield_surrogate import N_STATIC_SITE, YieldSurrogateModel
from models.yield_surrogate_v2 import YieldSurrogateV2


def _dummy_batch(
    b: int = 2,
    t: int = 365,
    f: int = 11,
    s: int = N_STATIC_SITE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    climate = torch.randn(b, t, f) * 0.05
    climate[..., 10] = 415.0
    static = torch.randn(b, s)
    static[:, 0] = 140.0
    region_id = torch.zeros(b, dtype=torch.long)
    return climate, static, region_id


def test_v2_forward_shape() -> None:
    model = YieldSurrogateV2()
    climate, static, region_id = _dummy_batch()
    y = model(climate, static, region_id)
    assert y.shape == (2,)


def test_v2_grad_smoke() -> None:
    model = YieldSurrogateV2()
    climate, static, region_id = _dummy_batch(b=1)
    climate.requires_grad_(True)
    y = model(climate, static, region_id)
    y.sum().backward()
    assert climate.grad is not None


def test_from_v1_checkpoint_synthetic() -> None:
    v1 = YieldSurrogateModel()
    v1.eval()
    path = Path("/tmp/test_yield_v1_synth.pt")
    torch.save(v1.state_dict(), path)
    v2 = YieldSurrogateV2.from_v1_checkpoint(path)
    v1_loaded = YieldSurrogateModel()
    v1_loaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    v1_loaded.eval()
    climate, static, region_id = _dummy_batch(b=1)
    with torch.no_grad():
        delta = v2.pape.delta(climate, region_id)
        assert float(delta.abs().max()) < 1e-6
        torch.manual_seed(42)
        y1 = v1_loaded(climate, static)
        torch.manual_seed(42)
        y2 = v2(climate, static, region_id)
        assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(
    not Path("models/yield_surrogate_v1.pt").is_file(),
    reason="v1 checkpoint not present",
)
def test_from_v1_checkpoint_repo() -> None:
    v2 = YieldSurrogateV2.from_v1_checkpoint("models/yield_surrogate_v1.pt")
    climate, static, region_id = _dummy_batch(b=1)
    with torch.no_grad():
        assert float(v2.pape.delta(climate, region_id).abs().max()) < 1e-6
