"""Tests for Phenology-Aware Positional Encoding (PAPE)."""

from __future__ import annotations

import numpy as np
import torch

from models.pape import (
    STAGE_NAMES,
    PhenologyAwarePositionalEncoding,
    crop_stage_one_hot,
    load_phenology_config,
    region_to_id,
)


def test_pape_deterministic() -> None:
    pape = PhenologyAwarePositionalEncoding(11)
    climate = torch.randn(2, 30, 11)
    region_id = torch.tensor([0, 1], dtype=torch.long)
    doy = torch.arange(1, 31).unsqueeze(0).expand(2, -1)
    d1 = pape.delta(climate, region_id, doy=doy)
    d2 = pape.delta(climate, region_id, doy=doy)
    assert torch.allclose(d1, d2)


def test_pape_param_budget() -> None:
    pape = PhenologyAwarePositionalEncoding(11)
    assert pape.count_parameters() < 50_000


def test_pape_zero_init() -> None:
    pape = PhenologyAwarePositionalEncoding(11)
    climate = torch.randn(1, 10, 11)
    region_id = torch.zeros(1, dtype=torch.long)
    delta = pape.delta(climate, region_id)
    assert float(delta.detach().abs().max()) < 1e-8


def test_crop_stage_doy_300_ghana_civ() -> None:
    setting_idx = STAGE_NAMES.index("main_crop_setting")
    for region in ("ghana", "civ"):
        oh = crop_stage_one_hot(region, 300)
        assert int(np.argmax(oh)) == setting_idx


def test_region_to_id_order() -> None:
    assert region_to_id("ghana") == 0
    assert region_to_id("civ") == 1
