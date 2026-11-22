from __future__ import annotations

import numpy as np


def test_bridge_hands_and_pose():
    from scripts.convert_openhands import openhands_frame_to_168
    kp = np.zeros((75, 3), dtype=np.float64)
    kp[0:33, :] = 0.5
    kp[0, :] = 0.4
    kp[33:54, :] = 0.6
    kp[54:75, :] = 0.7
    out = openhands_frame_to_168(kp)
    assert out.shape == (168,)
    assert out.dtype == np.float32
    assert out[63] == 1.0 and out[127] == 1.0 and out[167] == 1.0


def test_bridge_missing_hands_gives_zero_presence():
    from scripts.convert_openhands import openhands_frame_to_168
    kp = np.zeros((75, 3), dtype=np.float64)
    kp[0:33, :] = 0.5
    kp[0, :] = 0.4
    out = openhands_frame_to_168(kp)
    assert out[63] == 0.0 and out[127] == 0.0
    assert out[167] == 1.0
