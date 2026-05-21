"""Tests for TerraMind TiM facade."""

from __future__ import annotations

import pytest


def test_tim_modalities_validation() -> None:
    import torch

    from models.terramind_tim import TerraMindTiM

    model = TerraMindTiM(pretrained=False)
    x = {
        "S2L2A": torch.randn(1, 12, 32, 32),
        "S1GRD": torch.randn(1, 2, 32, 32),
        "DEM": torch.randn(1, 2, 32, 32),
    }
    out = model.predict(x, tim_modalities=["LULC", "NDVI"])
    assert out.dim() == 4

    with pytest.raises(ValueError, match="tim_modalities"):
        model.predict(x, tim_modalities=["FOO"])


def test_tim_predict_smoke() -> None:
    import torch

    from models.terramind_tim import TerraMindTiM

    model = TerraMindTiM(pretrained=False)
    x = {"S2L2A": torch.randn(2, 12, 16, 16)}
    out = model.predict(x, tim_modalities=["NDVI"])
    assert out.shape[0] == 2
