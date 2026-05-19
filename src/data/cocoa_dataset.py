"""
TorchGeo datasets for cocoa plantation mapping (Sentinel-1/2 + land-cover masks).

Pairs multi-band imagery GeoTIFFs with class masks (0=other, 1=full-sun cocoa,
2=agroforestry cocoa) for Prithvi / TerraTorch fine-tuning workflows.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from pyproj import CRS
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import IntersectionDataset, RasterDataset, random_bbox_assignment
from torchgeo.datasets.utils import Sample
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.samplers.utils import _to_tuple

# Land-cover class IDs in mask GeoTIFFs
CLASS_OTHER = 0
CLASS_FULL_SUN_COCOA = 1
CLASS_AGROFORESTRY_COCOA = 2

CLASS_NAMES: dict[int, str] = {
    CLASS_OTHER: "other",
    CLASS_FULL_SUN_COCOA: "full_sun_cocoa",
    CLASS_AGROFORESTRY_COCOA: "agroforestry_cocoa",
}

# Bands expected from Sentinel composite export (see sentinel_composite.py)
DEFAULT_IMAGERY_BANDS: tuple[str, ...] = (
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDVI",
    "EVI",
    "S1_VV",
    "S1_VH",
)

# Prithvi-friendly RGB subset for plotting
RGB_BANDS: tuple[str, ...] = ("B4", "B3", "B2")


class CocoaImagery(RasterDataset):
    """Sentinel-1/2 (or similar) multi-band imagery tiles."""

    filename_glob = "*.tif"
    filename_regex = r"^.*\.tif$"
    is_image = True
    separate_files = False
    all_bands = DEFAULT_IMAGERY_BANDS
    rgb_bands = RGB_BANDS


class CocoaMask(RasterDataset):
    """Per-pixel land-cover class masks aligned to imagery tiles."""

    filename_glob = "*.tif"
    filename_regex = r"^.*\.tif$"
    is_image = False
    separate_files = False
    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        CLASS_OTHER: (180, 180, 180, 255),
        CLASS_FULL_SUN_COCOA: (139, 69, 19, 255),
        CLASS_AGROFORESTRY_COCOA: (34, 139, 34, 255),
    }

    def __getitem__(self, index: Any) -> Sample:
        sample = super().__getitem__(index)
        mask = sample["mask"].long().squeeze()
        if mask.numel() > 0:
            unique = torch.unique(mask)
            valid = torch.tensor(list(CLASS_NAMES.keys()), device=mask.device)
            if not torch.isin(unique, valid).all():
                invalid = unique[~torch.isin(unique, valid)].tolist()
                raise ValueError(
                    f"Mask contains invalid class IDs {invalid}. "
                    f"Expected {list(CLASS_NAMES.keys())}."
                )
        sample["mask"] = mask
        return sample


class CocoaDataset(IntersectionDataset):
    """
    Spatially aligned Sentinel imagery and cocoa land-cover masks.

    Combines :class:`CocoaImagery` and :class:`CocoaMask` (both
    :class:`~torchgeo.datasets.RasterDataset`) via
    :class:`~torchgeo.datasets.IntersectionDataset` so samples share CRS,
    resolution, and geographic bounds.

    Imagery and masks must use matching filenames (e.g. ``tile_001.tif``) in
    separate directories.
    """

    def __init__(
        self,
        image_paths: Path | str = "data/processed/images",
        mask_paths: Path | str = "data/processed/masks",
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
    ) -> None:
        """
        Args:
            image_paths: Directory of multi-band imagery GeoTIFFs.
            mask_paths: Directory of single-band class mask GeoTIFFs (same basenames).
            crs: CRS to warp both modalities to (defaults to imagery CRS).
            res: Pixel size in CRS units (defaults to imagery resolution).
            bands: Imagery bands to load (defaults to all).
            transforms: Optional transform applied to the combined sample.
            cache: Cache rasterio file handles.
        """
        image_paths = Path(image_paths)
        mask_paths = Path(mask_paths)
        _verify_paired_tiles(image_paths, mask_paths)

        self.image = CocoaImagery(
            paths=image_paths,
            crs=crs,
            res=res,
            bands=bands,
            cache=cache,
        )
        self.mask = CocoaMask(
            paths=mask_paths,
            crs=self.image.crs,
            res=self.image.res,
            cache=cache,
        )
        super().__init__(self.image, self.mask, transforms=transforms)

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)


def _list_geotiffs(root: Path) -> list[Path]:
    pattern = os.path.join(str(root), "**", "*.tif")
    return sorted(Path(p) for p in glob.glob(pattern, recursive=True))


def _verify_paired_tiles(image_dir: Path, mask_dir: Path) -> None:
    """Ensure every imagery tile has a mask with the same basename."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Imagery directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_files = _list_geotiffs(image_dir)
    if not image_files:
        raise FileNotFoundError(f"No imagery GeoTIFFs matching *.tif in {image_dir}")

    missing: list[str] = []
    for image_path in image_files:
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            missing.append(image_path.name)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Missing mask GeoTIFFs for {len(missing)} imagery file(s): "
            f"{preview}{suffix}"
        )


class CocoaDataModule(GeoDataModule):
    """
    PyTorch Lightning DataModule for cocoa plantation segmentation.

    Uses random geographic bounding-box splits and TorchGeo samplers:
    - Train: random 224×224 patches (configurable)
    - Val/Test: gridded patches
    """

    def __init__(
        self,
        image_paths: Path | str = "data/processed/images",
        mask_paths: Path | str = "data/processed/masks",
        batch_size: int = 4,
        patch_size: int | tuple[int, int] = 224,
        length: int = 500,
        num_workers: int = 0,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        bands: Sequence[str] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            image_paths: Directory of imagery GeoTIFFs.
            mask_paths: Directory of mask GeoTIFFs.
            batch_size: Training batch size.
            patch_size: Crop size (height, width) in pixels — default 224 for Prithvi.
            length: Training samples per epoch.
            num_workers: DataLoader worker processes.
            train_val_test_split: Fractions for geographic bbox assignment.
            bands: Imagery bands to load.
            seed: RNG seed for reproducible train/val/test splits.
        """
        self.image_paths = Path(image_paths)
        self.mask_paths = Path(mask_paths)
        self.train_val_test_split = train_val_test_split
        self.seed = seed
        self.bands = bands

        super().__init__(
            CocoaDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

        patch_tuple = _to_tuple(self.patch_size)
        self.train_aug = K.AugmentationSequential(
            K.RandomResizedCrop(patch_tuple, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None},
                DataKey.IMAGE: {"resample": Resample.BILINEAR, "align_corners": None},
            },
        )
        self.aug = K.AugmentationSequential(data_keys=None, keepdim=True)

    def setup(self, stage: str) -> None:
        """Create datasets and samplers for the requested stage."""
        dataset = CocoaDataset(
            image_paths=self.image_paths,
            mask_paths=self.mask_paths,
            bands=self.bands,
        )
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_bbox_assignment(
            dataset,
            list(self.train_val_test_split),
            generator,
        )

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
        """Apply random geospatial augmentations during training."""
        if self.trainer and self.trainer.training and self.train_aug is not None:
            return self.train_aug(batch)
        return batch

