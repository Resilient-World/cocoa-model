"""Hard-example mining for AgriFM training (top-30% loss emphasis)."""

from __future__ import annotations

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.utils.data import WeightedRandomSampler


class HardExampleMiningCallback(Callback):
    """
    Track per-batch training loss and emphasize the top ``hard_fraction`` next epoch.

    For TorchGeo geo samplers, applies a loss multiplier on hard batches. For index
    samplers (synthetic mode), rebuilds a :class:`WeightedRandomSampler`.
    """

    def __init__(
        self,
        hard_fraction: float = 0.30,
        hard_weight: float = 3.0,
        easy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.hard_fraction = hard_fraction
        self.hard_weight = hard_weight
        self.easy_weight = easy_weight
        self._epoch_losses: list[float] = []
        self._sample_weights: np.ndarray | None = None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | dict[str, torch.Tensor] | None,
        batch: object,
        batch_idx: int,
    ) -> None:
        if outputs is None:
            return
        loss = outputs if torch.is_tensor(outputs) else outputs.get("loss")
        if loss is not None:
            self._epoch_losses.append(float(loss.detach().item()))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._epoch_losses:
            return
        losses = np.array(self._epoch_losses, dtype=np.float64)
        threshold = float(np.quantile(losses, 1.0 - self.hard_fraction))
        pl_module.hard_loss_threshold = threshold  # type: ignore[attr-defined]
        hard_frac = float(np.mean(losses >= threshold))
        trainer.logger.log_metrics(
            {"train/hard_fraction": hard_frac, "train/hard_loss_threshold": threshold},
            step=trainer.global_step,
        )
        self._epoch_losses.clear()

    def build_sample_weights(self, per_sample_losses: np.ndarray) -> WeightedRandomSampler:
        """Build sampler for synthetic/indexed datasets."""
        threshold = float(np.quantile(per_sample_losses, 1.0 - self.hard_fraction))
        weights = np.where(
            per_sample_losses >= threshold,
            self.hard_weight,
            self.easy_weight,
        ).astype(np.float64)
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
        )


class IndexHardMiningSampler:
    """Weighted index sampler updated each epoch from per-sample losses."""

    def __init__(self, num_samples: int, *, hard_fraction: float = 0.30) -> None:
        self.num_samples = num_samples
        self.hard_fraction = hard_fraction
        self.weights = torch.ones(num_samples, dtype=torch.float64)

    def set_losses(self, losses: np.ndarray) -> None:
        if len(losses) != self.num_samples:
            raise ValueError(f"Expected {self.num_samples} losses, got {len(losses)}")
        threshold = float(np.quantile(losses, 1.0 - self.hard_fraction))
        hard = torch.from_numpy(losses.astype(np.float64)) >= threshold
        self.weights = torch.where(hard, torch.tensor(3.0), torch.tensor(1.0)).double()

    def sampler(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.num_samples,
            replacement=True,
        )

    def __len__(self) -> int:
        return self.num_samples
