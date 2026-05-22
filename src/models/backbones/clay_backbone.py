"""Clay v1.5 geospatial encoder (Apache-2.0, made-with-clay/Clay)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

CLAY_HF_REPO = "made-with-clay/Clay-v1.5"
DEFAULT_EMBED_DIM = 384


class _StubClayEncoder(nn.Module):
    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(14, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ClayBackbone(nn.Module):
    def __init__(
        self, *, freeze: bool = True, use_hf: bool = True, cache_dir: str | Path | None = None
    ) -> None:
        super().__init__()
        self.embed_dim = DEFAULT_EMBED_DIM
        self._hf: nn.Module | None = None
        self._stub = _StubClayEncoder(self.embed_dim)
        if use_hf:
            try:
                from transformers import AutoModel

                self._hf = AutoModel.from_pretrained(
                    CLAY_HF_REPO,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    trust_remote_code=True,
                )
            except Exception:
                self._hf = None
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _fuse(self, batch_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        s2 = batch_dict["s2"].mean(dim=1).permute(0, 3, 1, 2)
        s1 = batch_dict["s1"].mean(dim=1).permute(0, 3, 1, 2)
        dem = batch_dict["dem"].permute(0, 3, 1, 2)
        return torch.cat([s2[:, :10], s1, dem], dim=1)

    def forward(self, batch_dict: dict[str, torch.Tensor | None]) -> torch.Tensor:
        assert batch_dict["s2"] is not None
        x = self._fuse({k: v for k, v in batch_dict.items() if v is not None})  # type: ignore[arg-type]
        if self._hf is not None:
            out = self._hf(pixel_values=x)
            feats = getattr(out, "last_hidden_state", out[0] if isinstance(out, tuple) else out)
            if feats.ndim == 3:
                n = int(feats.shape[1] ** 0.5)
                feats = feats.transpose(1, 2).reshape(feats.shape[0], -1, n, n)
            return feats
        return self._stub(x)

    def encode_parcel(self, parcel_inputs: dict[str, torch.Tensor | None]) -> torch.Tensor:
        return self.forward(parcel_inputs).mean(dim=(-2, -1))


__all__ = ["CLAY_HF_REPO", "ClayBackbone"]
