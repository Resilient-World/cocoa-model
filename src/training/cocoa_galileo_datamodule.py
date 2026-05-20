"""TorchGeo DataModule: cocoa tiles → Galileo multimodal batch dicts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import Sample

from data.cocoa_dataset import (
    CLASS_NAMES,
    CLASS_OTHER,
    CocoaDataModule,
    CocoaDataset,
    CocoaImagery,
    CocoaMask,
    DEFAULT_IMAGERY_BANDS,
)
from data.sentinel_composite import S2_OPTICAL_BANDS

# Galileo / sentinel_composite band groupings
GALILEO_S2_BANDS: tuple[str, ...] = tuple(S2_OPTICAL_BANDS)
GALILEO_S1_BANDS: tuple[str, ...] = ("S1_VV", "S1_VH")
GALILEO_NDVI_BAND: str = "NDVI"

IGNORE_MASK_VALUE = 255  # optional nodata in mask GeoTIFFs


class _AuxRaster(RasterDataset):
    """Single-modality auxiliary GeoTIFF stack (aligned filenames with imagery)."""

    is_image = True
    separate_files = False
    filename_glob = "*.tif"
    filename_regex = r"^.*\.tif$"

    def __init__(self, key: str, paths: Path | str, bands: Sequence[str] | None = None, **kwargs: Any) -> None:
        self.key = key
        self.all_bands = tuple(bands) if bands is not None else ()
        super().__init__(paths=paths, bands=bands, **kwargs)


class CocoaGalileoDataset(CocoaDataset):
    """
    Cocoa imagery + mask with optional aligned auxiliary stacks for Galileo.

    Auxiliary directories must use the same ``*.tif`` basenames as imagery tiles.
    Missing optional modalities are omitted from samples; the Galileo encoder
    receives zeros with mask=1 via :func:`data.utils.cocoa_batch_to_galileo_input`.
    """

    def __init__(
        self,
        image_paths: Path | str = "data/processed/images",
        mask_paths: Path | str = "data/processed/masks",
        *,
        srtm_paths: Path | str | None = None,
        era5_paths: Path | str | None = None,
        terraclim_paths: Path | str | None = None,
        dynamic_world_paths: Path | str | None = None,
        world_cereal_paths: Path | str | None = None,
        crs: Any = None,
        res: float | tuple[float, float] | None = None,
        bands: Sequence[str] | None = None,
        transforms: Any = None,
        cache: bool = True,
    ) -> None:
        super().__init__(
            image_paths=image_paths,
            mask_paths=mask_paths,
            crs=crs,
            res=res,
            bands=bands,
            transforms=transforms,
            cache=cache,
        )
        self._aux_rasters: dict[str, _AuxRaster] = {}
        aux_specs: list[tuple[str, Path | str | None, Sequence[str] | None]] = [
            ("srtm", srtm_paths, ("elevation", "slope")),
            ("era5", era5_paths, ("precip", "temperature_2m")),
            ("terraclim", terraclim_paths, ("aet", "def", "soil")),
            ("dynamic_world", dynamic_world_paths, None),  # 9-band probs
            ("world_cereal", world_cereal_paths, ("ag_class",)),
        ]
        for key, paths, band_names in aux_specs:
            if paths is None:
                continue
            self._aux_rasters[key] = _AuxRaster(
                key=key,
                paths=Path(paths),
                bands=band_names,
                crs=self.image.crs,
                res=self.image.res,
                cache=cache,
            )

    def __getitem__(self, index: Any) -> Sample:
        sample = super().__getitem__(index)
        for key, raster in self._aux_rasters.items():
            aux = raster[index]
            sample[key] = aux["image"]
        return sample


def _band_indices(source_bands: Sequence[str], selected: Sequence[str]) -> list[int]:
    return [list(source_bands).index(name) for name in selected]


def split_imagery_tensor(
    image: torch.Tensor,
    source_bands: Sequence[str],
) -> dict[str, torch.Tensor]:
    """
    Split a TorchGeo imagery tensor ``[C, H, W]`` or ``[B, C, H, W]`` into Galileo keys.
    """
    batched = image.dim() == 4
    if not batched:
        image = image.unsqueeze(0)

    s2_idx = _band_indices(source_bands, GALILEO_S2_BANDS)
    s1_idx = _band_indices(source_bands, GALILEO_S1_BANDS)
    ndvi_idx = _band_indices(source_bands, (GALILEO_NDVI_BAND,))

    s2 = image[:, s2_idx, ...].permute(0, 2, 3, 1).unsqueeze(1)  # B,T,H,W,C
    s1 = image[:, s1_idx, ...].permute(0, 2, 3, 1).unsqueeze(1)
    ndvi = image[:, ndvi_idx, ...].squeeze(1).unsqueeze(1)  # B,T,H,W

    return {
        "s2": s2,
        "s1": s1,
        "ndvi": ndvi,
    }


def sample_to_galileo_batch(
    sample: Sample,
    source_bands: Sequence[str],
) -> dict[str, torch.Tensor | None]:
    """Convert a TorchGeo sample into a Galileo ``batch_dict`` (single-item batch)."""
    image = sample["image"]
    if image.dim() == 3:
        parts = split_imagery_tensor(image, source_bands)
    else:
        parts = split_imagery_tensor(image, source_bands)

    batch: dict[str, torch.Tensor | None] = dict(parts)

    for key in ("srtm", "era5", "terraclim", "dynamic_world", "world_cereal", "location"):
        if key not in sample:
            continue
        tensor = sample[key]
        if not torch.is_tensor(tensor):
            continue
        if key == "srtm" and tensor.dim() == 3:
            batch["srtm"] = tensor.permute(1, 2, 0)  # H,W,C
        elif key in ("era5", "terraclim") and tensor.dim() == 3:
            # C,T,H or C,H,W — treat as T=1 spatial map → pool to T,C
            if tensor.shape[0] <= 8:
                batch[key] = tensor.permute(1, 0).mean(dim=-1) if tensor.dim() == 3 else tensor
            else:
                batch[key] = tensor.flatten(1).mean(-1).unsqueeze(0)
        elif key == "dynamic_world" and tensor.dim() == 3:
            batch["dynamic_world"] = tensor.permute(1, 2, 0)
        elif key == "world_cereal" and tensor.dim() == 3:
            batch["world_cereal"] = tensor.permute(1, 2, 0)[..., :1]
        else:
            batch[key] = tensor

    return batch


def collated_to_galileo_batch(
    batch: Sample,
    source_bands: Sequence[str],
) -> dict[str, torch.Tensor | None]:
    """Convert a collated TorchGeo batch (B,C,H,W) into Galileo inputs with batch dim."""
    image = batch["image"]
    b = image.shape[0]
    galileo: dict[str, torch.Tensor | None] = {}

    split = split_imagery_tensor(image, source_bands)
    galileo.update(split)

    for key in ("srtm", "era5", "terraclim", "dynamic_world", "world_cereal"):
        if key not in batch:
            continue
        tensor = batch[key]
        if key == "srtm":
            galileo["srtm"] = tensor.permute(0, 2, 3, 1)  # B,H,W,C
        elif key in ("era5", "terraclim"):
            if tensor.dim() == 4:
                galileo[key] = tensor.mean(dim=-1).mean(dim=-1)  # fallback
            elif tensor.dim() == 3:
                galileo[key] = tensor.permute(0, 2, 1) if tensor.shape[1] <= 8 else tensor
        elif key == "dynamic_world":
            galileo["dynamic_world"] = tensor.permute(0, 2, 3, 1)
        elif key == "world_cereal":
            galileo["world_cereal"] = tensor.permute(0, 2, 3, 1)[..., :1]

    _ = b  # batch size implied by image
    return galileo


def prepare_mask(batch: Sample, ignore_index: int = CLASS_OTHER) -> torch.Tensor:
    """Return ``[B, H, W]`` long mask; remap nodata and presence-only background to *ignore_index*."""
    mask = batch["mask"].long()
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    mask = mask.clone()
    mask[mask == IGNORE_MASK_VALUE] = ignore_index
    return mask


class CocoaGalileoDataModule(CocoaDataModule):
    """CocoaDataModule that emits Galileo multimodal tensors for segmentation training."""

    def __init__(
        self,
        *args: object,
        source_bands: Sequence[str] | None = None,
        srtm_paths: Path | str | None = None,
        era5_paths: Path | str | None = None,
        terraclim_paths: Path | str | None = None,
        dynamic_world_paths: Path | str | None = None,
        world_cereal_paths: Path | str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.source_bands = tuple(source_bands or self.bands or DEFAULT_IMAGERY_BANDS)
        self.aux_paths = {
            "srtm": Path(srtm_paths) if srtm_paths else None,
            "era5": Path(era5_paths) if era5_paths else None,
            "terraclim": Path(terraclim_paths) if terraclim_paths else None,
            "dynamic_world": Path(dynamic_world_paths) if dynamic_world_paths else None,
            "world_cereal": Path(world_cereal_paths) if world_cereal_paths else None,
        }

    def setup(self, stage: str) -> None:
        dataset = CocoaGalileoDataset(
            image_paths=self.image_paths,
            mask_paths=self.mask_paths,
            bands=self.bands,
            srtm_paths=self.aux_paths["srtm"],
            era5_paths=self.aux_paths["era5"],
            terraclim_paths=self.aux_paths["terraclim"],
            dynamic_world_paths=self.aux_paths["dynamic_world"],
            world_cereal_paths=self.aux_paths["world_cereal"],
        )
        generator = torch.Generator().manual_seed(self.seed)
        from torchgeo.datasets import random_bbox_assignment

        self.train_dataset, self.val_dataset, self.test_dataset = random_bbox_assignment(
            dataset,
            list(self.train_val_test_split),
            generator,
        )

        from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                cast(Any, self.train_dataset),
                self.patch_size,
                self.batch_size,
                self.length,
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                cast(Any, self.val_dataset),
                self.patch_size,
                self.patch_size,
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                cast(Any, self.test_dataset),
                self.patch_size,
                self.patch_size,
            )

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        batch["galileo"] = collated_to_galileo_batch(batch, self.source_bands)
        batch["mask"] = prepare_mask(batch)
        return batch

    @property
    def class_names(self) -> list[str]:
        return [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)]
