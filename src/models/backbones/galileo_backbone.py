"""
Galileo ViT encoder backbone for cocoa segmentation (A/B vs Prithvi/TerraTorch).

Weights are loaded lazily on the first forward pass from Hugging Face
``nasaharvest/galileo``. The encoder implementation uses the vendored
:mod:`models.backbones.vendor.single_file_galileo` so we do not require the full Galileo
repository layout at runtime (optional ``galileo`` pip package supplies
:data:`data.utils` helpers).
"""

from __future__ import annotations

import structlog

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from data.utils import cocoa_batch_to_galileo_input
from .vendor.single_file_galileo import Encoder as GalileoEncoder

log = structlog.get_logger(__name__)

GALILEO_HF_REPO = "nasaharvest/galileo"
SUPPORTED_MODEL_SIZES = frozenset({"nano", "tiny", "base"})

# Cocoa plantation classes (matches TorchGeo cocoa_dataset labels)
CLASS_OTHER = 0
CLASS_FULL_SUN = 1
CLASS_AGROFORESTRY = 2
DEFAULT_NUM_CLASSES = 3


def download_galileo_weights(
    model_size: str = "base",
    cache_dir: str | Path | None = None,
) -> Path:
    """Download Galileo checkpoint files into the Hugging Face cache."""
    if model_size not in SUPPORTED_MODEL_SIZES:
        raise ValueError(f"model_size must be one of {sorted(SUPPORTED_MODEL_SIZES)}")
    local = snapshot_download(
        repo_id=GALILEO_HF_REPO,
        allow_patterns=[f"models/{model_size}/*"],
        cache_dir=cache_dir,
    )
    return Path(local) / "models" / model_size


def _batch_masked(masked: Any, device: torch.device) -> Any:
    return type(masked)(
        **{
            k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
            for k, v in masked._asdict().items()
        }
    )


def _space_time_feature_map(
    s_t_x: torch.Tensor,
    s_t_m: torch.Tensor,
) -> torch.Tensor:
    """
    Collapse space-time tokens to a 2D feature map ``[B, D, H', W']``.

    Averages over time and channel-group dimensions, weighting by valid (unmasked) tokens.
    """
    valid = (1.0 - s_t_m).unsqueeze(-1)
    weighted = s_t_x * valid
    denom = valid.sum(dim=(3, 4)).clamp(min=1e-6)
    feat = weighted.sum(dim=(3, 4)) / denom
    return feat.permute(0, 3, 1, 2).contiguous()


class GalileoCocoaBackbone(nn.Module):
    """
    Frozen (by default) Galileo encoder producing dense patch features for segmentation.

    Parameters
    ----------
    model_size:
        Hugging Face folder name under ``models/`` (``nano``, ``tiny``, ``base``).
    freeze:
        If True, encoder weights are not trained.
    patch_size:
        Galileo flexi patch size (4 recommended for production; 8 for lighter compute).
    normalize:
        Apply Galileo pretraining min–max normalization in :func:`data.utils.cocoa_batch_to_galileo_input`.
    cache_dir:
        Optional Hugging Face hub cache directory.
  """

    def __init__(
        self,
        model_size: str = "base",
        freeze: bool = True,
        patch_size: int = 4,
        *,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if model_size not in SUPPORTED_MODEL_SIZES:
            raise ValueError(f"model_size must be one of {sorted(SUPPORTED_MODEL_SIZES)}")
        self.model_size = model_size
        self.freeze = freeze
        self.patch_size = patch_size
        self.normalize = normalize
        self.cache_dir = cache_dir
        self._device = torch.device(device or "cpu")

        self._encoder: GalileoEncoder | None = None
        self._embedding_size: int | None = None

    @property
    def embedding_size(self) -> int:
        if self._embedding_size is None:
            raise RuntimeError("Call forward() once to load weights before accessing embedding_size")
        return self._embedding_size

    @property
    def encoder(self) -> GalileoEncoder:
        """Loaded Galileo encoder (triggers lazy weight download on first access)."""
        return self._ensure_encoder(self._device)

    def set_freeze(self, freeze: bool) -> None:
        """Toggle backbone gradient flow (used for two-stage fine-tuning)."""
        self.freeze = freeze
        if self._encoder is None:
            return
        for param in self._encoder.parameters():
            param.requires_grad = not freeze
        if freeze:
            self._encoder.eval()
        else:
            self._encoder.train()

    def _ensure_encoder(self, device: torch.device) -> GalileoEncoder:
        if self._encoder is not None:
            return self._encoder

        weights_dir = download_galileo_weights(self.model_size, cache_dir=self.cache_dir)
        log.info("Loading Galileo-%s encoder from %s", self.model_size, weights_dir)
        encoder = GalileoEncoder.load_from_folder(weights_dir, device=device)
        self._embedding_size = encoder.embedding_size
        if self.freeze:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        self._encoder = encoder
        return encoder

    def _encode_masked(self, masked: Any, device: torch.device) -> torch.Tensor:
        encoder = self._ensure_encoder(device)
        batched = _batch_masked(masked, device)
        encoded = encoder(
            batched.space_time_x,
            batched.space_x,
            batched.time_x,
            batched.static_x,
            batched.space_time_mask,
            batched.space_mask,
            batched.time_mask,
            batched.static_mask,
            batched.months,
            self.patch_size,
        )
        s_t_x, _, _, _, s_t_m, _, _, _, _ = encoded
        return _space_time_feature_map(s_t_x, s_t_m)

    def forward(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        """
        Parameters
        ----------
        batch_dict:
            See :func:`data.utils.cocoa_batch_to_galileo_input`.

        Returns
        -------
        torch.Tensor
            Patch feature map ``[B, D, H/P, W/P]``.
        """
        device = self._infer_device(batch_dict)
        masked = cocoa_batch_to_galileo_input(batch_dict, normalize=self.normalize)
        return self._encode_masked(masked, device)

    def encode_parcel(self, parcel_inputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
        """
        Global-pooled embedding for one FTW parcel tile batch.

        Returns
        -------
        torch.Tensor
            Shape ``[B, D]``.
        """
        feat_map = self.forward(parcel_inputs)
        return feat_map.mean(dim=(-2, -1))

    def _infer_device(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.device:
        for value in batch_dict.values():
            if torch.is_tensor(value):
                return value.device
        return self._device


class _GalileoLiteFPNHead(nn.Module):
    """Lightweight FPN-style head for a single-scale Galileo feature map."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        mid = max(in_channels // 2, 32)
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        if self.patch_size > 1:
            logits = F.interpolate(
                logits,
                scale_factor=float(self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits


class GalileoSegmentation(nn.Module):
    """
    Galileo backbone + UPerNet-style decoder for cocoa plantation segmentation.

    Classes default to 3: other, full-sun cocoa, agroforestry cocoa.
    """

    def __init__(
        self,
        model_size: str = "base",
        num_classes: int = DEFAULT_NUM_CLASSES,
        patch_size: int = 4,
        freeze_backbone: bool = True,
        *,
        decoder: str = "upernet",
        decoder_channels: int = 256,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        if decoder not in {"upernet", "fpn"}:
            raise ValueError("decoder must be 'upernet' or 'fpn'")
        self.decoder_name = decoder
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.backbone = GalileoCocoaBackbone(
            model_size=model_size,
            freeze=freeze_backbone,
            patch_size=patch_size,
            normalize=normalize,
            cache_dir=cache_dir,
        )
        self._decoder: nn.Module | None = None

    def set_backbone_freeze(self, freeze: bool) -> None:
        """Freeze or unfreeze the Galileo encoder weights."""
        self.backbone.set_freeze(freeze)

    def _ensure_decoder(self, embed_dim: int, device: torch.device) -> nn.Module:
        if self._decoder is not None:
            return self._decoder

        if self.decoder_name == "fpn":
            self._decoder = _GalileoLiteFPNHead(embed_dim, self.num_classes, self.patch_size)
        else:
            from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder

            depth = 4
            self._decoder = UPerNetDecoder(
                encoder_channels=[embed_dim] * depth,
                encoder_depth=depth,
                decoder_channels=self.decoder_channels,
            )
            self._upernet_depth = depth
        self._decoder = self._decoder.to(device)
        return self._decoder

    def _decoder_forward(self, feat: torch.Tensor) -> torch.Tensor:
        decoder = self._decoder
        assert decoder is not None
        if self.decoder_name == "fpn":
            return decoder(feat)
        # Build a coarse-to-fine pyramid from the single Galileo map
        pyramid = [feat]
        for _ in range(self._upernet_depth - 1):
            pyramid.append(F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2))
        pyramid = pyramid[::-1]
        logits = decoder(pyramid)
        if self.patch_size > 1:
            logits = F.interpolate(
                logits,
                scale_factor=float(self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def forward(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Segmentation logits ``[B, num_classes, H, W]``.
        """
        features = self.backbone(batch_dict)
        device = features.device
        decoder = self._ensure_decoder(self.backbone.embedding_size, device)
        return self._decoder_forward(features)
