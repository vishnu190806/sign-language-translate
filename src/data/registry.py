from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd


class SignLanguageDataset(ABC):
    @abstractmethod
    def download(self, dest_dir: str | Path) -> None: ...

    @abstractmethod
    def build_metadata(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_label_map(self) -> Dict[str, int]: ...

    @abstractmethod
    def get_splits(self) -> Dict[str, List[str]]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class DatasetRegistry:
    _registry: Dict[str, type[SignLanguageDataset]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(dataset_cls: type[SignLanguageDataset]):
            cls._registry[name] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[SignLanguageDataset]:
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(f"Dataset '{name}' not registered. Available: {available}")
        return cls._registry[name]

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry.keys())
