from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.htma import HTMAClassifier


class DPCPretrainer(nn.Module):
    NUM_WINDOWS = 7
    WINDOW_LEN = 10
    CONTEXT_WINDOWS = 4
    PRED_STEPS = 3

    def __init__(self, model: HTMAClassifier, gru_hidden: int = 256):
        super().__init__()
        self.model = model
        dim = model.config.embedding_dim
        self.gru = nn.GRU(dim, gru_hidden, batch_first=True)
        self.predictors = nn.ModuleList(
            [nn.Linear(gru_hidden, dim) for _ in range(self.PRED_STEPS)]
        )

    def encode_window(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        h = m.input_proj(x)
        h = m.input_norm(h)
        h = m.appe(h)
        rope_cos, rope_sin = m._get_rope(h)
        for block in m.blocks:
            h = block(h, rope_cos, rope_sin, mask)
        return m.pool(h, mask)

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        B, W, T, _ = windows.shape
        return self.encode_window(windows.reshape(B * W, T, -1)).view(
            B, W, self.model.config.embedding_dim)

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = windows.shape[0]
        z = self.encode_windows(windows)
        _, hidden = self.gru(z[:, :self.CONTEXT_WINDOWS])
        c = hidden.squeeze(0)
        losses = []
        hits = 0.0
        for s in range(self.PRED_STEPS):
            pred = self.predictors[s](c)
            tgt = z[:, self.CONTEXT_WINDOWS + s]
            scores = pred @ tgt.t()
            logp = F.log_softmax(scores, dim=1)
            losses.append(-logp.diag().mean())
            hits += (scores.argmax(1) == torch.arange(B, device=scores.device)).float().mean()
        return torch.stack(losses).mean(), hits / self.PRED_STEPS
