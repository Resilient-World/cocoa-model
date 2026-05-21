"""
MIT-licensed Video Swin Transformer 3D with synchronized spatiotemporal downsampling.

Architecture follows Li et al. (RSE 2026; arXiv:2505.21357) Section 2.2.1 and the
AgriFM reference implementation (Apache-2.0). Only tensor math is reimplemented here;
pretrained weights are loaded separately in :mod:`models.agrifm_backbone`.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, reduce as einops_reduce
from timm.layers import DropPath, trunc_normal_


def _window_partition(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)


def _window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    B: int,
    D: int,
    H: int,
    W: int,
) -> torch.Tensor:
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


def _get_window_size(
    x_size: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | tuple[tuple[int, int, int], ...]:
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


def _compute_mask(
    D: int,
    H: int,
    W: int,
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = _window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        idx = self.relative_position_index[:N, :N].reshape(-1)
        rel_bias = self.relative_position_bias_table[idx].reshape(N, N, -1).permute(2, 0, 1)
        attn = attn + rel_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int, int] = (2, 7, 7),
        shift_size: tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward_part1(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        window_size, shift_size = _get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = _window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *window_size, C)
        shifted_x = _window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    """Synchronized spatiotemporal downsampling via mean over temporal groups."""

    def __init__(
        self,
        dim: int,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample_step: tuple[int, int, int] = (2, 2, 2),
        mean_frame_down: bool = False,
    ) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.downsample_step = downsample_step
        self.mean_frame_down = mean_frame_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        D_step, H_step, W_step = self.downsample_step
        if D_step >= 2 and self.mean_frame_down and D % D_step != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, D_step - D % D_step))
        h = 0 if H_step == 1 else 1
        w = 0 if W_step == 1 else 1
        x0 = x[:, ::D_step, 0::H_step, 0::W_step, :]
        x1 = x[:, ::D_step, h::H_step, 0::W_step, :]
        x2 = x[:, ::D_step, 0::H_step, w::W_step, :]
        x3 = x[:, ::D_step, h::H_step, w::W_step, :]
        if D_step >= 2 and self.mean_frame_down:
            for i in range(1, D_step):
                x0 = x0 + x[:, i::D_step, 0::H_step, 0::W_step, :]
                x1 = x1 + x[:, i::D_step, h::H_step, 0::W_step, :]
                x2 = x2 + x[:, i::D_step, 0::H_step, w::W_step, :]
                x3 = x3 + x[:, i::D_step, h::H_step, w::W_step, :]
            x0, x1, x2, x3 = x0 / D_step, x1 / D_step, x2 / D_step, x3 / D_step
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: tuple[int, int, int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
        downsample_step: tuple[int, int, int] = (2, 2, 2),
        mean_frame_down: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if i % 2 == 0 else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            downsample(
                dim=dim,
                norm_layer=norm_layer,
                downsample_step=downsample_step,
                mean_frame_down=mean_frame_down,
            )
            if downsample is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        window_size, shift_size = _get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = _compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        return rearrange(x, "b d h w c -> b c d h w")


class SwinPatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (4, 2, 2),
        in_chans: int = 10,
        embed_dim: int = 128,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Video tensor ``[B, C, T, H, W]`` (cocoa-model contract).
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            d, wh, ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
        return x


class SwinTransformer3D(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        depths: tuple[int, ...] = (2, 2, 18, 2),
        num_heads: tuple[int, ...] = (4, 8, 16, 32),
        window_size: tuple[int, int, int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample_steps: tuple[tuple[int, int, int], ...] = (
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ),
        mean_frame_down: bool = True,
        use_checkpoint: bool = False,
        feature_fusion: str = "cat",
        reduce_feature_scale: int | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.feature_fusion = feature_fusion
        self.mean_frame_down = mean_frame_down
        self.reduce_feature_scale = reduce_feature_scale
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                BasicLayer(
                    dim=int(embed_dim * 2**i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                    downsample_step=downsample_steps[i_layer],
                    mean_frame_down=mean_frame_down,
                    use_checkpoint=use_checkpoint,
                )
            )
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)

    def _collect_feat(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_fusion == "cat":
            feat = (
                einops_reduce(x, "b c (t s) h w -> b c t h w", "mean", s=self.reduce_feature_scale)
                if self.reduce_feature_scale is not None
                else x
            )
            b, c, d, h, w = feat.shape
            return torch.reshape(feat, (b, c * d, h, w))
        if self.feature_fusion == "mean":
            return torch.mean(x, dim=2)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        out_feats: list[torch.Tensor] = []
        if 0 in self.out_indices:
            out_feats.append(self._collect_feat(x))
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous())
            if i + 1 in self.out_indices:
                out_feats.append(self._collect_feat(x))
        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)
        x = rearrange(x, "n d h w c -> n c d h w")
        encoder_features = self._collect_feat(x)
        return {"encoder_features": encoder_features, "features_list": out_feats}


class PretrainingSwinTransformer3DEncoder(nn.Module):
    """Patch embedding + Video Swin backbone (AgriFM pretraining encoder)."""

    def __init__(
        self,
        patch_embed: SwinPatchEmbed3D,
        backbone: SwinTransformer3D,
    ) -> None:
        super().__init__()
        self.patch_embed = patch_embed
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.patch_embed(x)
        return self.backbone(x)


def build_agrifm_encoder(
    in_chans: int = 10,
    embed_dim: int = 128,
    depths: tuple[int, ...] = (2, 2, 18, 2),
    num_heads: tuple[int, ...] = (4, 8, 16, 32),
    patch_size: tuple[int, int, int] = (4, 2, 2),
    window_size: tuple[int, int, int] = (8, 7, 7),
    mean_frame_down: bool = True,
) -> PretrainingSwinTransformer3DEncoder:
    """Factory matching AgriFM S2 cropland_mapping defaults."""
    patch_embed = SwinPatchEmbed3D(
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )
    backbone = SwinTransformer3D(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        out_indices=(0, 1, 2, 3),
        mean_frame_down=mean_frame_down,
    )
    return PretrainingSwinTransformer3DEncoder(patch_embed, backbone)
