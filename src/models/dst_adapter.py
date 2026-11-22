from __future__ import annotations

import torch
import torch.nn as nn

from src.models.htma import HTMAClassifier


class SpatialAdapter(nn.Module):
    def __init__(self, dim: int, adapter_dim: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(dim, adapter_dim)
        self.conv = nn.Conv2d(
            adapter_dim, adapter_dim,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=adapter_dim,
        )
        self.norm = nn.LayerNorm(adapter_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = x.transpose(1, 2).unsqueeze(-1)
        x = self.conv(x)
        x = x.squeeze(-1).transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        x = self.up_proj(x)
        return residual + x


class TemporalAdapter(nn.Module):
    def __init__(self, dim: int, adapter_dim: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(dim, adapter_dim)
        self.conv = nn.Conv2d(
            adapter_dim, adapter_dim,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=adapter_dim,
        )
        self.norm = nn.LayerNorm(adapter_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = x.transpose(1, 2).unsqueeze(-1)
        x = self.conv(x)
        x = x.squeeze(-1).transpose(1, 2)
        x = self.norm(x)
        x = self.act(x)
        x = self.up_proj(x)
        return residual + x


class ChannelAwareFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, backbone: torch.Tensor, adapter_out: torch.Tensor) -> torch.Tensor:
        diff = adapter_out - backbone
        pooled = self.gap(diff.transpose(1, 2)).squeeze(-1)
        weights = self.mlp(pooled)
        return backbone + weights.unsqueeze(1) * diff


class DSTAdapter(nn.Module):
    def __init__(self, model: HTMAClassifier, adapter_dim: int = 64):
        super().__init__()
        self.model = model
        dim = model.config.embedding_dim
        num_blocks = model.config.num_blocks

        for param in self.model.parameters():
            param.requires_grad = False

        self.spatial_adapters = nn.ModuleList(
            [SpatialAdapter(dim, adapter_dim) for _ in range(num_blocks)]
        )
        self.temporal_adapters = nn.ModuleList(
            [TemporalAdapter(dim, adapter_dim) for _ in range(num_blocks)]
        )
        self.spatial_fusions = nn.ModuleList(
            [ChannelAwareFusion(dim) for _ in range(num_blocks)]
        )
        self.temporal_fusions = nn.ModuleList(
            [ChannelAwareFusion(dim) for _ in range(num_blocks)]
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        model = self.model
        x = model.input_proj(features)
        x = model.input_norm(x)
        x = model.appe(x)

        rope_cos, rope_sin = model._get_rope(x)

        for i, block in enumerate(model.blocks):
            b = block
            x = b.norm1(x)
            x = b.conv_mixer(x)
            x = x + b.dropout(b.attention(x, rope_cos, rope_sin, mask))

            x = self.spatial_fusions[i](x, self.spatial_adapters[i](x))

            x = b.norm2(x)
            x = x + b.dropout(b.ffn(x))

            x = self.temporal_fusions[i](x, self.temporal_adapters[i](x))

            x = b.norm3(x)

        x = model.pool(x, mask)
        return model.head(x)

    def count_parameters(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
