from __future__ import annotations

import torch.nn as nn

from src.config import ModelConfig
from src.models.htma import HTMAClassifier, HTMAConfig


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "htma":
        htm_config = HTMAConfig(
            input_dim=config.input_dim,
            embedding_dim=config.embedding_dim,
            num_blocks=config.htma_blocks,
            num_heads=config.num_heads,
            dropout=config.dropout,
            num_classes=config.num_classes,
            fusion_method=getattr(config, "fusion_method", "early"),
            use_separate_encoders=getattr(config, "use_separate_encoders", False),
            max_seq_len=getattr(config, "max_seq_len", 128),
        )
        return HTMAClassifier(htm_config)
    else:
        raise ValueError(f"Unknown model: {config.name}")
