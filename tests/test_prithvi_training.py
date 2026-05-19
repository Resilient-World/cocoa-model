"""Unit tests for Prithvi training helpers."""

import torch

from data.cocoa_dataset import DEFAULT_IMAGERY_BANDS
from training.cocoa_prithvi_datamodule import (
    PRITHVI_RGB_BANDS,
    PRITHVI_SENTINEL2_BANDS,
    CocoaPrithviDataModule,
    hls_bands_for_input,
    prithvi_normalization_tensors,
    resolve_band_indices,
)
from training.train_prithvi_cocoa import build_prithvi_model_args


def test_resolve_band_indices_rgb_subset() -> None:
    source = ("B2", "B3", "B4", "B8", "B11", "B12", "S1_VV")
    indices = resolve_band_indices(source, PRITHVI_RGB_BANDS)
    assert indices == [2, 1, 0]


def test_prithvi_normalization_shapes() -> None:
    mean, std = prithvi_normalization_tensors(6)
    assert mean.shape == (1, 6, 1, 1)
    assert std.shape == (1, 6, 1, 1)
    mean3, std3 = prithvi_normalization_tensors(3)
    assert mean3.shape == (1, 3, 1, 1)


def test_build_model_args_num_classes() -> None:
    args = build_prithvi_model_args(
        num_classes=3,
        backbone_bands=hls_bands_for_input(PRITHVI_SENTINEL2_BANDS),
        backbone_in_chans=6,
    )
    assert args["num_classes"] == 3
    assert args["decoder"] == "UperNetDecoder"
    assert args["backbone_in_chans"] == 6
    assert len(args["necks"]) == 3


def test_normalize_batch_channels() -> None:
    indices = resolve_band_indices(DEFAULT_IMAGERY_BANDS, PRITHVI_SENTINEL2_BANDS)
    image = torch.rand(2, len(DEFAULT_IMAGERY_BANDS), 224, 224)[:, indices]
    mean, std = prithvi_normalization_tensors(6)
    normalized = (image - mean) / std
    assert normalized.shape == (2, 6, 224, 224)
