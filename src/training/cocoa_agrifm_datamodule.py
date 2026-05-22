"""TorchGeo DataModule: cocoa tiles → AgriFM ``[B,C,T,H,W]`` tensors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import kornia.augmentation as K
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets.utils import Sample

from data.cocoa_dataset import CLASS_OTHER, DEFAULT_IMAGERY_BANDS, CocoaDataModule
from data.sentinel_composite import S2_OPTICAL_BANDS
from training.hard_example_mining import IndexHardMiningSampler

AGRI_FM_S2_BANDS: tuple[str, ...] = tuple(S2_OPTICAL_BANDS)

# AgriFM cropland_mapping S2 normalization (reflectance scale 0–1 after /10000)
AGRI_FM_S2_MEAN = torch.tensor(
    [
        4179.19,
        4065.91,
        3957.27,
        5207.45,
        4327.12,
        4873.16,
        5049.16,
        5111.08,
        3056.86,
        2490.97,
    ],
    dtype=torch.float32,
)
AGRI_FM_S2_STD = torch.tensor(
    [
        4041.52,
        3691.00,
        3629.33,
        2973.52,
        3569.73,
        3085.92,
        2937.56,
        2806.04,
        1808.30,
        1694.20,
    ],
    dtype=torch.float32,
)
REFLECTANCE_SCALE = 10_000.0


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


def binary_cocoa_mask(mask: torch.Tensor) -> torch.Tensor:
    """``[B,H,W]`` or ``[B,1,H,W]`` long mask → ``[B,1,H,W]`` float cocoa presence."""
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    cocoa = (mask != CLASS_OTHER).float()
    return cocoa.unsqueeze(1)


def imagery_to_agrifm_tensor(
    image: torch.Tensor,
    band_indices: list[int],
    *,
    num_frames: int,
    reflectance_scale: float = REFLECTANCE_SCALE,
    temporal_mode: str = "repeat_augment",
) -> torch.Tensor:
    """
    Build ``[B, C, T, H, W]`` from ``[B, C_src, H, W]``.

    ``repeat_augment`` duplicates the patch with independent noise per frame.
    """
    x = image[:, band_indices, ...] / reflectance_scale
    mean = (AGRI_FM_S2_MEAN / reflectance_scale).view(1, -1, 1, 1).to(x.device)
    std = (AGRI_FM_S2_STD / reflectance_scale).view(1, -1, 1, 1).to(x.device)
    x = (x - mean) / std.clamp(min=1e-6)

    b, c, h, w = x.shape
    if temporal_mode == "repeat_augment":
        frames = []
        for _ in range(num_frames):
            noise = torch.randn_like(x) * 0.02
            frames.append((x + noise).unsqueeze(2))
        return torch.cat(frames, dim=2)
    raise ValueError(f"Unknown temporal_mode: {temporal_mode}")


class SyntheticAgriFMDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory synthetic patches for CI / offline training smoke tests."""

    def __init__(
        self,
        n_samples: int,
        *,
        patch_size: int = 64,
        num_frames: int = 8,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        n = n_samples
        _t, h, w, c = num_frames, patch_size, patch_size, len(AGRI_FM_S2_BANDS)
        images = rng.normal(0.2, 0.05, (n, c, h, w)).astype(np.float32)
        labels = (rng.random((n, 1, h, w)) < 0.35).astype(np.float32)
        self._images = torch.from_numpy(images)
        self._labels = torch.from_numpy(labels)
        self.num_frames = num_frames

    def __len__(self) -> int:
        return self._images.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = self._images[idx]
        frames = torch.stack(
            [img + torch.randn_like(img) * 0.02 for _ in range(self.num_frames)], dim=1
        )
        return {"agrifm": frames, "target": self._labels[idx]}


class CocoaAgriFMDataModule(CocoaDataModule):
    """
    Cocoa patches → AgriFM video tensor + binary cocoa target.

    Subsets 10-band S2 and stacks temporal frames (repeat+aug or Zarr timeseries).
    """

    def __init__(
        self,
        *args: object,
        input_bands: Sequence[str] | None = None,
        source_bands: Sequence[str] | None = None,
        num_frames: int = 8,
        temporal_mode: str = "repeat_augment",
        timeseries_zarr: Path | str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.source_bands = tuple(source_bands or self.bands or DEFAULT_IMAGERY_BANDS)
        self.input_bands = tuple(input_bands or AGRI_FM_S2_BANDS)
        self.reflectance_scale = REFLECTANCE_SCALE
        self._band_indices = resolve_band_indices(self.source_bands, self.input_bands)
        self.num_frames = max(3, min(32, num_frames))
        self.temporal_mode = temporal_mode
        self.timeseries_zarr = Path(timeseries_zarr) if timeseries_zarr else None
        self._frame_aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        image = batch["image"]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        agrifm = imagery_to_agrifm_tensor(
            image,
            self._band_indices,
            num_frames=self.num_frames,
            reflectance_scale=self.reflectance_scale,
            temporal_mode=self.temporal_mode,
        )
        if self.trainer and self.trainer.training:
            b, c, t, h, w = agrifm.shape
            flat = agrifm.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            aug = self._frame_aug({"image": flat})["image"]
            agrifm = aug.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        batch["agrifm"] = agrifm
        mask = batch.get("mask")
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            batch["target"] = binary_cocoa_mask(mask.long())
        return batch


class SyntheticAgriFMDataModule(pl.LightningDataModule):
    """Lightweight datamodule for ``--synthetic`` training runs."""

    def __init__(
        self,
        n_train: int = 80,
        n_val: int = 20,
        *,
        batch_size: int = 4,
        patch_size: int = 64,
        num_frames: int = 8,
        seed: int = 0,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.n_train = n_train
        self.n_val = n_val
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.seed = seed
        self.num_workers = num_workers
        self.train_dataset: SyntheticAgriFMDataset | None = None
        self.val_dataset: SyntheticAgriFMDataset | None = None
        self.hard_sampler: IndexHardMiningSampler | None = None

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate"):
            self.train_dataset = SyntheticAgriFMDataset(
                self.n_train,
                patch_size=self.patch_size,
                num_frames=self.num_frames,
                seed=self.seed,
            )
            self.val_dataset = SyntheticAgriFMDataset(
                self.n_val,
                patch_size=self.patch_size,
                num_frames=self.num_frames,
                seed=self.seed + 1,
            )
            self.hard_sampler = IndexHardMiningSampler(self.n_train)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        sampler = self.hard_sampler.sampler() if self.hard_sampler else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def update_hard_weights(self, per_sample_losses: np.ndarray) -> None:
        if self.hard_sampler is not None:
            self.hard_sampler.set_losses(per_sample_losses)
