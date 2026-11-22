from __future__ import annotations

import torch
import torch.nn as nn

from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

_SKIP_BUFFERS = {"appe.pe", "rope.cos_cached", "rope.sin_cached"}


class HTMACTCModel(nn.Module):
    def __init__(self, checkpoint: str, n_glosses: int, dropout: float = 0.3,
                 device: str | torch.device = "cpu", train_encoder: bool = False):
        super().__init__()
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        htm = HTMAConfig(num_classes=263, max_seq_len=512)
        self.encoder = MultiHeadHTMAClassifier(htm, sources=tuple(ckpt.get("sources", ["include"])))
        state = {k: v for k, v in ckpt["model_state_dict"].items() if k not in _SKIP_BUFFERS}
        self.encoder.load_state_dict(state, strict=False)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if train_encoder:
            for name, param in self.encoder.named_parameters():
                if "blocks." in name and ".norm" not in name and "rope" not in name:
                    param.requires_grad = True
            self.encoder.train()
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_glosses + 1),
        )
        with torch.no_grad():
            self.head[-1].bias.zero_()
            self.head[-1].bias[0] = -2.0
        self.n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder.input_proj(x)
        h = self.encoder.input_norm(h)
        h = self.encoder.appe(h)
        cos, sin = self.encoder._get_rope(h)
        for block in self.encoder.blocks:
            h = block(h, cos, sin, mask)
        return self.head(h)
