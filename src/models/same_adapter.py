from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.htma import HTMAClassifier


class MoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        bottleneck_dim: int,
        num_signer_features: int = 6,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, bottleneck_dim),
                    nn.ReLU(),
                    nn.Linear(bottleneck_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(num_signer_features, num_experts)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, signer_features: torch.Tensor) -> torch.Tensor:
        gate_weights = F.softmax(self.gate(signer_features), dim=-1)

        expert_out = torch.zeros_like(x)
        for w, expert in zip(gate_weights.unbind(-1), self.experts):
            expert_out = expert_out + w.view(-1, 1, 1) * expert(x)

        alpha = torch.sigmoid(self.alpha)
        return alpha * expert_out + (1 - alpha) * x


class SAMEAdapter(nn.Module):
    NUM_SIGNER_FEATURES = 6

    def __init__(
        self,
        model: HTMAClassifier,
        num_experts: int = 4,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.model = model
        dim = model.config.embedding_dim
        num_blocks = model.config.num_blocks

        for param in self.model.parameters():
            param.requires_grad = False

        self.moe_layers = nn.ModuleList(
            [
                MoELayer(dim, num_experts, bottleneck_dim, self.NUM_SIGNER_FEATURES)
                for _ in range(num_blocks)
            ]
        )

    def _extract_signer_features(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == 168, f"Signer stats assume fused 168-d layout, got {x.size(-1)}"
        x_pooled = x.mean(dim=1)
        left_hand = x_pooled[:, 0:64]
        right_hand = x_pooled[:, 64:128]
        pose = x_pooled[:, 128:168]
        return torch.stack(
            [
                left_hand.mean(dim=-1),
                left_hand.std(dim=-1).clamp(min=1e-6),
                right_hand.mean(dim=-1),
                right_hand.std(dim=-1).clamp(min=1e-6),
                pose.mean(dim=-1),
                pose.std(dim=-1).clamp(min=1e-6),
            ],
            dim=-1,
        )

    def _head_for(self, source: str | None):
        heads = getattr(self.model, "heads", None)
        if source is not None and heads is not None and source in heads:
            return heads[source]
        return self.model.head

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None,
        source: str | None = None,
    ) -> torch.Tensor:
        model = self.model
        x = model.input_proj(features)
        x = model.input_norm(x)
        x = model.appe(x)

        rope_cos, rope_sin = model._get_rope(x)
        signer_features = self._extract_signer_features(features)

        for block, moe in zip(model.blocks, self.moe_layers):
            x = block(x, rope_cos, rope_sin, mask)
            x = moe(x, signer_features)

        x = model.pool(x, mask)
        return self._head_for(source)(x)

    def snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.moe_layers.state_dict().items()}

    def reset(self, snapshot: dict | None = None) -> None:
        if snapshot is not None:
            self.moe_layers.load_state_dict(snapshot)
        else:
            for layer in self.moe_layers:
                layer.reset_parameters() if hasattr(layer, "reset_parameters") else None
            for m in self.moe_layers.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def adapt(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        source: str | None = None,
        num_steps: int = 5,
        lr: float = 3e-5,
        w_ent: float = 1.0,
        w_mcc: float = 1.0,
        w_kl: float = 1.0,
    ) -> torch.Tensor:
        model_device = next(self.model.parameters()).device
        assert features.device == model_device, (
            f"Feature/model device mismatch: {features.device} vs {model_device}")
        self.model.eval()

        with torch.no_grad():
            source_logits = self.forward(features, mask, source=source)
            source_probs = F.softmax(source_logits, dim=-1)

        params = [p for p in self.moe_layers.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)

        logits = source_logits
        for _ in range(num_steps):
            optimizer.zero_grad()
            logits = self.forward(features, mask, source=source)
            probs = F.softmax(logits, dim=-1)

            log_probs = torch.log(probs + 1e-8)
            entropy_loss = -torch.mean(torch.sum(probs * log_probs, dim=-1))

            q = probs**2
            q = q / (q.sum(dim=0, keepdim=True) + 1e-8)
            mcc_loss = -torch.sum(torch.sqrt(probs * q + 1e-8))

            kl_loss = F.kl_div(
                log_probs,
                source_probs,
                reduction="batchmean",
            )

            loss = w_ent * entropy_loss + w_mcc * mcc_loss + w_kl * kl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

        return logits.detach()
