"""DataModule wrapper: TorchGeo cocoa patches → Prithvi-ready tensors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from terratorch.datasets import HLSBands
from torchgeo.datasets.utils import Sample

from data.cocoa_dataset import (
    CLASS_NAMES,
    DEFAULT_IMAGERY_BANDS,
    CocoaDataModule,
)

# Sentinel-2 band names aligned with Prithvi EO v2 pretraining (6 channels)
PRITHVI_SENTINEL2_BANDS: tuple[str, ...] = ("B2", "B3", "B4", "B8", "B11", "B12")

# Standard RGB-only subset (3 channels) — set input_bands to this tuple to fine-tune with RGB
PRITHVI_RGB_BANDS: tuple[str, ...] = ("B4", "B3", "B2")

# Prithvi-EO-2.0 normalization constants (HLS dynamic range; see TerraTorch prithvi_vit.py)
PRITHVI_V2_MEAN_6 = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
PRITHVI_V2_STD_6 = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)

HLS_BAND_ORDER_6: tuple[HLSBands, ...] = (
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
)

HLS_BAND_ORDER_RGB: tuple[HLSBands, ...] = (
    HLSBands.RED,
    HLSBands.GREEN,
    HLSBands.BLUE,
)


def resolve_band_indices(
    source_bands: Sequence[str],
    selected_bands: Sequence[str],
) -> list[int]:
    """Return channel indices to subset a multi-band tensor."""
    indices: list[int] = []
    for name in selected_bands:
        if name not in source_bands:
            raise ValueError(
                f"Band '{name}' not in dataset bands {list(source_bands)}. "
                "Load the band in CocoaImagery or adjust --input-bands."
            )
        indices.append(list(source_bands).index(name))
    return indices


def prithvi_normalization_tensors(
    num_channels: int,
    *,
    reflectance_scale: float = 10_000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-channel mean/std tensors for Prithvi v2.

    Assumes input reflectance is in [0, 1] and scales by reflectance_scale before
    applying HLS-style statistics (matches TerraTorch Prithvi fine-tuning examples).
    """
    if num_channels == 6:
        mean = PRITHVI_V2_MEAN_6
        std = PRITHVI_V2_STD_6
    elif num_channels == 3:
        mean = PRITHVI_V2_MEAN_6[:3]
        std = PRITHVI_V2_STD_6[:3]
    else:
        raise ValueError(
            f"Prithvi normalization supports 3 or 6 channels, got {num_channels}. "
            "Set --input-bands to 3 RGB or 6 Prithvi Sentinel-2 bands."
        )

    mean_t = torch.tensor(mean, device=device).view(1, num_channels, 1, 1) / reflectance_scale
    std_t = torch.tensor(std, device=device).view(1, num_channels, 1, 1) / reflectance_scale
    return mean_t, std_t


def hls_bands_for_input(input_bands: Sequence[str]) -> list[HLSBands]:
    """Map selected Sentinel band names to HLSBands for the Prithvi backbone."""
    if tuple(input_bands) == PRITHVI_RGB_BANDS:
        return list(HLS_BAND_ORDER_RGB)
    if tuple(input_bands) == PRITHVI_SENTINEL2_BANDS:
        return list(HLS_BAND_ORDER_6)
    raise ValueError(
        "input_bands must match PRITHVI_RGB_BANDS or PRITHVI_SENTINEL2_BANDS "
        f"(got {list(input_bands)})."
    )


class CocoaPrithviDataModule(CocoaDataModule):
    """
    CocoaDataModule that subsets bands and applies Prithvi v2 normalization.

    Use ``input_bands=PRITHVI_SENTINEL2_BANDS`` (6 channels, default) or
    ``input_bands=PRITHVI_RGB_BANDS`` (3 channels) when your tiles only include RGB.
    """

    def __init__(
        self,
        *args: object,
        input_bands: Sequence[str] | None = None,
        source_bands: Sequence[str] | None = None,
        reflectance_scale: float = 10_000.0,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.source_bands = tuple(source_bands or self.bands or DEFAULT_IMAGERY_BANDS)
        self.input_bands = tuple(input_bands or PRITHVI_SENTINEL2_BANDS)
        self.reflectance_scale = reflectance_scale
        self._band_indices = resolve_band_indices(self.source_bands, self.input_bands)
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def _normalize_prithvi(self, batch: Sample) -> Sample:
        image = batch["image"]
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image[:, self._band_indices, ...]

        if self._mean is None or self._std is None:
            self._mean, self._std = prithvi_normalization_tensors(
                len(self._band_indices),
                reflectance_scale=self.reflectance_scale,
                device=image.device,
            )

        batch["image"] = (image - self._mean) / self._std
        if "mask" in batch and batch["mask"].dim() == 4:
            batch["mask"] = batch["mask"].squeeze(1)
        return batch

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        return self._normalize_prithvi(batch)

    @property
    def num_input_channels(self) -> int:
        return len(self._band_indices)

    @property
    def prithvi_hls_bands(self) -> list[HLSBands]:
        return hls_bands_for_input(self.input_bands)

    @property
    def class_names(self) -> list[str]:
        return [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)]
