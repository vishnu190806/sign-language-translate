# Task 1.2: Geometry-Aware Augmentations

## What Was Implemented

Created `src/features/augment_geo.py` with the `GeometryAugmentor` class containing 6 geometry-preserving augmentations for sign language translation data:

| Augmentation | Description | Range | Applies To |
|---|---|---|---|
| `camera_depth_variation` | Scales z-coordinates to simulate depth variation | s ∈ [0.8, 1.2] | All landmarks (z only) |
| `hv_shift` | Adds uniform displacement to x,y of all landmarks | Δ ∈ [-0.15, 0.15] | All landmarks |
| `hand_size_variation` | Scales hand landmarks relative to wrist center | s ∈ [0.85, 1.15] | Hand landmarks only |
| `viewpoint_rotation` | 2D rotation of entire skeleton around origin | θ ∈ [-15°, 15°] | All landmarks (x,y) |
| `finger_articulation` | Rotates finger joints (PIP/DIP/TIP) around MCP | θ ∈ [-10°, 10°] | Hand landmarks only |
| `elbow_displacement` | Translates all hand landmarks by forearm vector | Δ ∈ [-0.1, 0.1] | Hand landmarks only |

Key design decisions:
- Follows existing `LandmarkAugmentor` pattern (rng parameter, copy-on-write)
- Each method is independently callable with configurable probability
- `__call__` chains all augmentations at 30% probability each
- Uses `numpy.random.Generator` for reproducibility via seed
- No comments added (per AGENTS.md)

## What Was Tested and Test Results

Created `tests/test_augment_geo.py` with **27 tests** across 8 test classes:

| Test Class | Tests | What's Verified |
|---|---|---|
| `TestCameraDepthVariation` | 4 | Shape, reproducibility, modification, z-only constraint |
| `TestHVShift` | 3 | Shape, reproducibility, modification |
| `TestHandSizeVariation` | 4 | Shape, reproducibility, hands-only constraint, wrist unchanged |
| `TestViewpointRotation` | 4 | Shape, reproducibility, zero-angle identity, 90° rotation matrix |
| `TestFingerArticulation` | 4 | Shape, reproducibility, zero-angle identity, modification |
| `TestElbowDisplacement` | 4 | Shape, reproducibility, hands-only constraint, modification |
| `TestGeometryAugmentorCall` | 4 | Shape, reproducibility, p=0 identity, p=1 modification |

**Result: 27/27 passed**

Full test suite: 104 passed, 16 failed (pre-existing failures from missing dataset files and `scripts.category_analysis` module — not introduced by this change).

## Files Changed

- `src/features/augment_geo.py` (new, 166 lines)
- `tests/test_augment_geo.py` (new, 200 lines)

## Self-Review Findings

- All augmentations correctly respect the feature vector layout: 128 hand dims (2×64) + 40 pose dims
- `hand_size_variation` and `finger_articulation` correctly leave pose landmarks untouched
- `camera_depth_variation` only modifies z-coordinates (indices i*3+2 within each chunk)
- `viewpoint_rotation` mathematically verified: 90° rotation produces x'=-y, y'=x
- `finger_articulation` uses per-finger pivot rotation around MCP/CMC joints for 5 finger chains
- The `_FakeRNG` helper in tests properly controls both `random()` (probability) and `uniform()` (parameter) calls

## Issues or Concerns

None. All acceptance criteria met:
- [x] All 6 augmentations implemented
- [x] Each augmentation preserves sign meaning (shape + reasonable ranges)
- [x] Unit tests pass (27/27)
- [x] Code follows existing patterns (LandmarkAugmentor)
- [x] No comments added
