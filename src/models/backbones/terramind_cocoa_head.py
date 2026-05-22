"""
UPerNet decoder head for TerraMind cocoa segmentation.

Reference: TerraMind-1.0-base PANGAEA UPerNet config (~59.57 mIoU). Full weight
parity may require TerraTorch ``terramind_v1_base_encdec`` decoder checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TerraMindCocoaUPerNetHead(nn.Module):
    """
    UPerNet-style decoder over a single TerraMind feature map.

    Builds a 4-level pyramid via successive pooling for
    ``segmentation_models_pytorch`` UPerNetDecoder.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        *,
        decoder_channels: int = 256,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self._decoder: nn.Module | None = None
        self._classifier: nn.Conv2d | None = None

    def _ensure_decoder(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._decoder is not None:
            return
        from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder

        depth = 4
        self._decoder = UPerNetDecoder(
            encoder_channels=[self.embed_dim] * depth,
            encoder_depth=depth,
            decoder_channels=self.decoder_channels,
        ).to(device=device, dtype=dtype)
        self._classifier = nn.Conv2d(self.decoder_channels, self.num_classes, kernel_size=1).to(
            device=device, dtype=dtype
        )

    @staticmethod
    def _build_pyramid(feat: torch.Tensor, depth: int = 4) -> list[torch.Tensor]:
        pyramid = [feat]
        cur = feat
        for _ in range(depth - 1):
            cur = F.avg_pool2d(cur, kernel_size=2, stride=2)
            pyramid.append(cur)
        return pyramid

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feat:
            Backbone map ``[B, D, H, W]``.

        Returns
        -------
        torch.Tensor
            Logits ``[B, num_classes, H, W]`` (upsampled to input resolution).
        """
        self._ensure_decoder(feat.device, feat.dtype)
        assert self._decoder is not None and self._classifier is not None
        h, w = feat.shape[-2:]
        pyramid = self._build_pyramid(feat)
        decoded = self._decoder(*pyramid)
        logits = self._classifier(decoded)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


__all__ = ["TerraMindCocoaUPerNetHead"]
