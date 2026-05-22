"""TorchGeo DataModule: cocoa tiles → TerraMind modality dicts."""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.cocoa_dataset import CLASS_OTHER, CocoaDataModule
from data.utils import cocoa_batch_to_terramind_input
from training.hard_example_mining import IndexHardMiningSampler


def binary_cocoa_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    cocoa = (mask != CLASS_OTHER).float()
    return cocoa.unsqueeze(1)


class SyntheticTerraMindDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic multimodal patches for TerraMind training smoke tests."""

    def __init__(self, n_samples: int, *, patch_size: int = 64, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        n, h, w = n_samples, patch_size, patch_size
        self._labels = torch.from_numpy((rng.random((n, 1, h, w)) < 0.35).astype(np.float32))
        self._s2 = torch.from_numpy(rng.normal(0.2, 0.05, (n, 10, h, w)).astype(np.float32))
        self._s1 = torch.from_numpy(rng.normal(-12, 2, (n, 2, h, w)).astype(np.float32))
        self._dem = torch.from_numpy(rng.normal(200, 50, (n, 2, h, w)).astype(np.float32))

    def __len__(self) -> int:
        return self._labels.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t, h, w = 4, self._labels.shape[2], self._labels.shape[3]
        s2 = self._s2[idx].permute(1, 2, 0).unsqueeze(0).expand(t, -1, -1, -1).unsqueeze(0)
        s1 = self._s1[idx].permute(1, 2, 0).unsqueeze(0).expand(t, -1, -1, -1).unsqueeze(0)
        dem = self._dem[idx].permute(1, 2, 0).unsqueeze(0)
        return {"s2": s2, "s1": s1, "dem": dem, "target": self._labels[idx]}


class CocoaTerraMindDataModule(CocoaDataModule):
    """Cocoa patches with TerraMind modality dict + binary target."""

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        s2 = batch["image"]
        if s2.dim() == 4:
            s2 = s2.unsqueeze(1)
        t = max(3, min(8, s2.shape[1] if s2.dim() == 5 else 1))
        if s2.dim() == 4:
            s2 = s2.unsqueeze(1).expand(-1, t, -1, -1, -1)
        s1 = torch.zeros(
            s2.shape[0], t, s2.shape[2], s2.shape[3], 2, device=s2.device, dtype=s2.dtype
        )
        dem = torch.zeros(
            s2.shape[0], s2.shape[2], s2.shape[3], 2, device=s2.device, dtype=s2.dtype
        )
        td_batch = {"s2": s2, "s1": s1, "srtm": dem}
        batch["terramind"] = cocoa_batch_to_terramind_input(td_batch)
        mask = batch.get("mask")
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            batch["target"] = binary_cocoa_mask(mask.long())
        return batch


class SyntheticTerraMindDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_train: int = 80,
        n_val: int = 20,
        *,
        batch_size: int = 4,
        patch_size: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_train = n_train
        self.n_val = n_val
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.seed = seed
        self.train_dataset: SyntheticTerraMindDataset | None = None
        self.val_dataset: SyntheticTerraMindDataset | None = None
        self.hard_sampler: IndexHardMiningSampler | None = None

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate"):
            self.train_dataset = SyntheticTerraMindDataset(
                self.n_train, patch_size=self.patch_size, seed=self.seed
            )
            self.val_dataset = SyntheticTerraMindDataset(
                self.n_val, patch_size=self.patch_size, seed=self.seed + 1
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
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def update_hard_weights(self, per_sample_losses: np.ndarray) -> None:
        if self.hard_sampler is not None:
            self.hard_sampler.set_losses(per_sample_losses)
