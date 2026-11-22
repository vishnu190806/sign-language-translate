from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.spec import FeatureSpec, DEFAULT_FEATURE_SPEC
from src.features.normalize import build_frame_features, normalize_hand_landmarks
from src.infer.buffer import RingBuffer
from src.infer.gate import StabilityGate
from src.infer.sentence import SentenceAssembler


@pytest.fixture
def spec():
    return DEFAULT_FEATURE_SPEC


@pytest.fixture
def dummy_hand(spec):
    return np.random.rand(spec.hand_landmarks, spec.hand_dims).astype(np.float32)


@pytest.fixture
def dummy_pose(spec):
    return np.random.rand(33, spec.pose_dims).astype(np.float32)


@pytest.fixture
def label_map():
    return {"hello": 0, "thank_you": 1, "yes": 2, "no": 3}
