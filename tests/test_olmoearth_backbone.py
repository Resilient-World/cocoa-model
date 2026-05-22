"""OlmoEarth backbone smoke tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load(name: str, rel: str):
    path = _SRC / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_olmoearth_encode_shape() -> None:
    bb_mod = _load("olmoearth_bb", "models/backbones/olmoearth_backbone.py")
    OlmoEarthBackbone = bb_mod.OlmoEarthBackbone
    bb = OlmoEarthBackbone(model_size="nano", use_hf=False)
    s2 = torch.randn(1, 4, 32, 32, 10)
    s1 = torch.randn(1, 4, 32, 32, 2)
    era5 = torch.randn(1, 4, 5)
    dem = torch.randn(1, 32, 32, 2)
    feat = bb.encode(s2, s1, era5, dem)
    assert feat.ndim == 4


def test_olmoearth_seg_forward() -> None:
    _load("olmoearth_bb", "models/backbones/olmoearth_backbone.py")
    head_mod = _load("olmoearth_head", "models/backbones/olmoearth_cocoa_head.py")
    # Minimal seg wiring without models package __init__
    bb_mod = sys.modules["olmoearth_bb"]
    OlmoEarthCocoaSegmentation = type(
        "OlmoEarthCocoaSegmentation",
        (),
        {
            "__init__": lambda self, model_size="base", use_hf=True: setattr(
                self,
                "backbone",
                bb_mod.OlmoEarthBackbone(model_size=model_size, use_hf=use_hf),
            )
            or setattr(
                self,
                "head",
                head_mod.OlmoEarthCocoaSegHead(
                    embed_dim=bb_mod.EMBED_DIM_BY_SIZE[model_size], out_size=(64, 64)
                ),
            ),
        },
    )
    model = OlmoEarthCocoaSegmentation(model_size="tiny", use_hf=False)
    batch = {
        "s2": torch.randn(1, 4, 64, 64, 10),
        "s1": torch.randn(1, 4, 64, 64, 2),
        "era5": torch.randn(1, 4, 5),
        "dem": torch.randn(1, 64, 64, 2),
        "location": torch.zeros(1, 2),
        "months": torch.tensor([[6, 7, 8, 9]]),
    }
    feats = model.backbone(batch)
    logits = model.head(feats)
    assert logits.shape[-2:] == (64, 64)
