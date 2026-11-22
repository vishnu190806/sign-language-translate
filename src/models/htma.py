from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import (
    AttentionPooling,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
)


@dataclass
class HTMAConfig:
    input_dim: int = 168
    embedding_dim: int = 256
    num_blocks: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    num_classes: int = 263
    max_seq_len: int = 128
    fusion_method: str = "early"
    use_separate_encoders: bool = False
    hand_input_dim: int = 128
    pose_input_dim: int = 40

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads


class Conv1DTemporalMixer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return residual + x


class MultiTokenConv(nn.Module):
    def __init__(self, num_heads: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            num_heads, num_heads,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=num_heads,
        )

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        return self.conv(attn_scores)


class HTMAAttention(nn.Module):
    def __init__(self, config: HTMAConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.k_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.v_proj = nn.Linear(config.embedding_dim, config.embedding_dim * 2)
        self.out_proj = nn.Linear(config.embedding_dim * 2, config.embedding_dim)

        self.mta = MultiTokenConv(config.num_heads)
        self.group_norm = nn.GroupNorm(config.num_heads, config.embedding_dim * 2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d * 2).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            seq_len = q.size(2)
            cos = rope_cos[:, :, :seq_len, :]
            sin = rope_sin[:, :, :seq_len, :]
            q = q * cos + self._rotate_half(q) * sin
            k = k * cos + self._rotate_half(k) * sin

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = self.mta(attn_scores)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = attn_out.transpose(1, 2)
        attn_out = self.group_norm(attn_out)
        attn_out = attn_out.transpose(1, 2)
        attn_out = self.out_proj(attn_out)

        return attn_out

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class HTMABlock(nn.Module):
    def __init__(self, config: HTMAConfig):
        super().__init__()
        self.conv_mixer = Conv1DTemporalMixer(config.embedding_dim)
        self.attention = HTMAAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
            nn.Dropout(config.dropout),
        )
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.norm3 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(x)
        x = self.conv_mixer(x)
        x = x + self.dropout(self.attention(x, rope_cos, rope_sin, mask))
        x = self.norm2(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm3(x)
        return x


class HandEncoder(nn.Module):
    def __init__(self, config: HTMAConfig):
        super().__init__()
        self.proj = nn.Linear(config.hand_input_dim, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class PoseEncoder(nn.Module):
    def __init__(self, config: HTMAConfig):
        super().__init__()
        self.proj = nn.Linear(config.pose_input_dim, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class ConcatFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([a, b], dim=-1))


class AttentionFusion(nn.Module):
    def __init__(self, config: HTMAConfig):
        super().__init__()
        d = config.embedding_dim
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.scale = d ** -0.5
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        B, T, D = a.shape
        q = self.q_proj(a)
        k = self.k_proj(b)
        v = self.v_proj(b)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.bmm(weights, v)
        return self.norm(a + out)


class GatingFusion(nn.Module):
    def __init__(self, config: HTMAConfig):
        super().__init__()
        d = config.embedding_dim
        self.gate_proj = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([a, b], dim=-1)))
        return self.norm(gate * a + (1 - gate) * b)


class HTMAClassifier(nn.Module):
    def __init__(self, config: HTMAConfig | None = None):
        super().__init__()
        if config is None:
            config = HTMAConfig()
        self.config = config

        self.use_separate_encoders = config.use_separate_encoders and config.fusion_method != "early"
        self.fusion_method = config.fusion_method

        if self.use_separate_encoders:
            self.hand_encoder = HandEncoder(config)
            self.pose_encoder = PoseEncoder(config)
            self.input_norm = nn.LayerNorm(config.embedding_dim)
        else:
            self.input_proj = nn.Linear(config.input_dim, config.embedding_dim)
            self.input_norm = nn.LayerNorm(config.embedding_dim)

        self.appe = SinusoidalPositionalEncoding(config.embedding_dim, config.max_seq_len)
        self.rope = RotaryPositionalEmbedding(config.head_dim, config.max_seq_len)
        self.rope_dim = config.embedding_dim // config.num_heads

        self.blocks = nn.ModuleList([
            HTMABlock(config) for _ in range(config.num_blocks)
        ])

        if self.fusion_method == "multiscale":
            self.scale_fusions = nn.ModuleList([
                ConcatFusion(config.embedding_dim) for _ in range(config.num_blocks)
            ])

        self.pool = AttentionPooling(config.embedding_dim)

        if self.use_separate_encoders and self.fusion_method == "late":
            self.fusion = ConcatFusion(config.embedding_dim)

        self.head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fusion_method == "multiscale":
            return self._forward_multiscale(x, mask)
        if self.use_separate_encoders:
            return self._forward_separate(x, mask)
        return self._forward_early(x, mask)

    def _forward_early(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.appe(x)

        rope_cos, rope_sin = self._get_rope(x)

        for block in self.blocks:
            x = block(x, rope_cos, rope_sin, mask)

        x = self.pool(x, mask)
        logits = self.head(x)
        return logits

    def _forward_separate(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        hand_feats = x[:, :, : self.config.hand_input_dim]
        pose_feats = x[:, :, self.config.hand_input_dim :]

        h = self.hand_encoder(hand_feats)
        p = self.pose_encoder(pose_feats)

        h = self.appe(h)
        p = self.appe(p)

        rope_cos, rope_sin = self._get_rope(h)

        for block in self.blocks:
            h = block(h, rope_cos, rope_sin, mask)
            p = block(p, rope_cos, rope_sin, mask)

        if self.fusion_method == "late":
            fused = self.fusion(h, p)
        else:
            fused = h + p

        fused = self.pool(fused, mask)
        logits = self.head(fused)
        return logits

    def _forward_multiscale(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        if self.use_separate_encoders:
            hand_feats = x[:, :, : self.config.hand_input_dim]
            pose_feats = x[:, :, self.config.hand_input_dim :]
            h = self.hand_encoder(hand_feats)
            p = self.pose_encoder(pose_feats)
            h = self.appe(h)
            p = self.appe(p)
        else:
            h = self.input_proj(x)
            h = self.input_norm(h)
            h = self.appe(h)
            p = h

        rope_cos, rope_sin = self._get_rope(h)

        for i, block in enumerate(self.blocks):
            h = block(h, rope_cos, rope_sin, mask)
            p = block(p, rope_cos, rope_sin, mask)
            if i < len(self.scale_fusions):
                h = self.scale_fusions[i](h, p)

        x = self.pool(h, mask)
        logits = self.head(x)
        return logits

    def _get_rope(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        dim = self.rope_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(t, inv_freq)
        cos_half = freqs.cos()
        sin_half = freqs.sin()
        cos = torch.repeat_interleave(cos_half, 2, dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.repeat_interleave(sin_half, 2, dim=-1).unsqueeze(0).unsqueeze(0)
        return cos, sin


class MultiHeadHTMAClassifier(HTMAClassifier):
    def __init__(
        self,
        config: HTMAConfig | None = None,
        sources: tuple[str, ...] = ("include", "isl500"),
        head_classes: dict[str, int] | None = None,
    ):
        super().__init__(config)
        self.sources = list(sources)
        hc = head_classes or {}
        self.heads = nn.ModuleDict({
            s: nn.Sequential(
                nn.Linear(config.embedding_dim, config.embedding_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.embedding_dim, hc.get(s, config.num_classes)),
            )
            for s in self.sources
        })
        if self.head[3].weight.shape == self.heads[self.sources[0]][3].weight.shape:
            self.heads[self.sources[0]].load_state_dict(self.head.state_dict())

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.appe(x)
        rope_cos, rope_sin = self._get_rope(x)
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin, mask)
        return self.pool(x, mask)

    def predict_all(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pooled = self.encode(x, mask)
        return {s: self.heads[s](pooled) for s in self.sources}

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        source: str | None = None,
    ) -> torch.Tensor:
        pooled = self.encode(x, mask)
        if source is None:
            return self.head(pooled)
        return self.heads[source](pooled)
