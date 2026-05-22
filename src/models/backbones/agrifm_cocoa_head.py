"""
Versatile multi-scale decoder for AgriFM cocoa segmentation.

Follows the fusion pattern in AgriFM (interpolate, concat, conv) with three
upsampling steps and a binary cocoa logit map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgriFMCocoaSegHead(nn.Module):
    """
    Multi-scale fusion head for binary cocoa probability maps.

    Parameters
    ----------
    embed_dim:
        Base channel width after each fusion block (AgriFM default 128).
    skip_channels:
        Channel widths for skip connections at each upsampling step
        (coarse-to-fine: stage_2, stage_3, stage_4 spatial means).
    out_size:
        Output spatial size ``(H, W)`` after bilinear upsampling.
    dropout_p:
        Optional dropout before the final conv (default 0 for CQR workflows).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        skip_channels: tuple[int, ...] = (256, 512, 1024),
        out_size: tuple[int, int] = (256, 256),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if len(skip_channels) != 3:
            raise ValueError("skip_channels must have length 3")
        self.embed_dim = embed_dim
        self.out_size = out_size
        self.fusion_blocks = nn.ModuleList()
        for skip_ch in skip_channels:
            in_ch = embed_dim + skip_ch
            self.fusion_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, embed_dim, kernel_size=3, padding=1),
                )
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.classifier = nn.Conv2d(embed_dim, 1, kernel_size=1)

    @staticmethod
    def _stage_to_bchw(stage: torch.Tensor) -> torch.Tensor:
        """Average over time and return ``[B,C,H,W]``."""
        mean = stage.mean(dim=1)
        return mean.permute(0, 3, 1, 2).contiguous()

    def forward(self, stages: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        stages:
            Backbone outputs ``stage_1`` … ``stage_4``.

        Returns
        -------
        torch.Tensor
            Logits ``[B, 1, H_out, W_out]``.
        """
        x = self._stage_to_bchw(stages["stage_1"])
        skip_keys = ("stage_2", "stage_3", "stage_4")
        for fusion, key in zip(self.fusion_blocks, skip_keys, strict=True):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            skip = F.interpolate(
                self._stage_to_bchw(stages[key]),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = fusion(torch.cat([x, skip], dim=1))
        x = self.out_conv(x)
        x = self.dropout(x)
        x = self.classifier(x)
        if x.shape[-2:] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        return x
