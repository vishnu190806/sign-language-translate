from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StabilityGateConfig:
    threshold: float = 0.50
    consecutive_frames: int = 12
    cooldown_frames: int = 30
    agreement_ratio: float = 0.60
    min_confidence_delta: float = 0.08


@dataclass
class ModelConfig:
    name: str = "htma"
    input_dim: int = 168
    embedding_dim: int = 256
    htma_blocks: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    num_classes: int = 263
    fusion_method: str = "early"
    use_separate_encoders: bool = False
    max_seq_len: int = 128


@dataclass
class ProcessingConfig:
    sequence_length: int = 32
    stability_gate: StabilityGateConfig = field(default_factory=StabilityGateConfig)


@dataclass
class TTSConfig:
    engine: str = "piper"
    voice: str = "en_US-lessac-medium"
    sample_rate: int = 22050
    length_scale: float = 1.0
    sentence_silence: float = 0.0
    voice_dir: str | None = None
    device: int | None = None
    max_queue_size: int = 1


@dataclass
class PathsConfig:
    assets: str = "assets"
    checkpoints: str = "models"
    logs: str = "logs"
    data_root: str = "data"
    raw: str = "data/raw"
    processed: str = "data/processed"


@dataclass
class DatasetConfig:
    name: str = "isl"
    root: str = "data/raw"
    processed: str = "data/processed"
    num_workers: int = 6
    prefetch_factor: int = 2
    persistent_workers: bool = True
    subset: str | None = None


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    early_stopping_patience: int = 15
    use_amp: bool = True
    seed: int = 42
    balanced_sampler: bool = False
    oversample_targets: str | None = None


@dataclass
class AugmentationConfig:
    mode: str = "basic"
    probs: dict[str, float] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        cfg = cls()
        if "dataset" in data:
            d = dict(data["dataset"])
            cfg.dataset = DatasetConfig(**{k: v for k, v in d.items() if k in DatasetConfig.__dataclass_fields__})
        if "model" in data:
            m = dict(data["model"])
            m.pop("stability_gate", None)
            cfg.model = ModelConfig(**{k: v for k, v in m.items() if k in ModelConfig.__dataclass_fields__})
        if "processing" in data:
            proc_data = data["processing"].copy()
            if "stability_gate" in proc_data:
                proc_data["stability_gate"] = StabilityGateConfig(**proc_data["stability_gate"])
            cfg.processing = ProcessingConfig(**proc_data)
        if "tts" in data:
            cfg.tts = TTSConfig(**data["tts"])
        if "paths" in data:
            cfg.paths = PathsConfig(**data["paths"])
        if "train" in data:
            cfg.train = TrainConfig(**data["train"])
        if "augmentation" in data:
            aug_data = dict(data["augmentation"])
            probs = aug_data.pop("probs", {}) or {}
            cfg.augmentation = AugmentationConfig(mode=aug_data.get("mode", "basic"), probs=dict(probs))
        return cfg

    def merge_yaml(self, path: str | Path) -> Config:
        with open(path, "r") as f:
            override = yaml.safe_load(f) or {}
        data = {
            "dataset": self.dataset.__dict__.copy(),
            "model": self.model.__dict__.copy(),
            "processing": {
                **self.processing.__dict__,
                "stability_gate": self.processing.stability_gate.__dict__,
            },
            "tts": self.tts.__dict__.copy(),
            "paths": self.paths.__dict__.copy(),
            "train": self.train.__dict__.copy(),
            "augmentation": {"mode": self.augmentation.mode, "probs": dict(self.augmentation.probs)},
        }

        def deep_merge(base: dict, override: dict) -> dict:
            for k, v in override.items():
                if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v
            return base

        merged = deep_merge(data, override)
        return Config._from_dict(merged)


def load_config(
    base_path: str | Path = "configs/default.yaml",
    override_path: str | Path | None = None,
) -> Config:
    cfg = Config.from_yaml(base_path)
    if override_path:
        cfg = cfg.merge_yaml(override_path)
    return cfg
