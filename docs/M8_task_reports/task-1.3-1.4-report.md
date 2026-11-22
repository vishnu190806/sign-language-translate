# Task 1.3-1.4: Temporal and Landmark-Specific Augmentations

## What Was Implemented

Added 7 new augmentation methods to the `LandmarkAugmentor` class in `src/features/augment.py` (grew from 137 to 321 lines), plus updated the existing `random_translation` to cover pose landmarks with the specified [-0.1, 0.1] displacement range.

| # | Method | Description | Default Prob |
|---|--------|-------------|-------------|
| 1 | `frame_permutation` | Shuffles frames in non-relevant segments (low landmark variation) | 0.2 |
| 2 | `speed_variation` | Resamples temporal dimension at 0.8x-1.2x speed via `np.linspace` index mapping | 0.2 |
| 3 | `random_frame_drop` | Drops 10-20% of frames (zeroed with presence flags cleared) | 0.3 |
| 4 | `spatial_rotation` | 3D rotation matrix (XYZ Euler, [-15°, 15°]) applied to all hand and pose landmarks | 0.2 |
| 5 | `random_translation` (updated) | Now shifts both hands AND pose landmarks with [-0.1, 0.1] displacement | 0.3 |
| 6 | `frame_fusion` | Weighted blending of each frame with its successor (alpha 0.3-0.7) | 0.2 |
| 7 | `random_keypoint_removal` | Randomly zeros 10-30% of individual feature dimensions; clears associated hand/pose presence flags | 0.2 |

All methods follow the existing patterns: optional `rng` parameter, `self.spec` for layout metadata, return `np.ndarray` of same shape.

## Tests

Added 21 new tests in `TestNewAugmentations` class in `tests/test_features.py`:

- **Shape preservation**: each new augmentation tested to confirm output shape equals input
- **Reproducibility**: same seed produces identical output
- **Input modification**: augmentations actually change the data (not identity)
- **Presence flag clearing**: `random_keypoint_removal` zeroes presence flags for affected hand/pose groups
- **Zero-probability identity**: high `variation_threshold` in `frame_permutation` leaves sequence unchanged
- **All-augmentations shape sweep**: bulk shape check across all 7 new methods

**Test results**: 29/29 in `test_features.py` passed. All 125 pre-existing passing tests remain green. 16 failures are pre-existing (missing dataset files / missing `scripts.category_analysis` module — unrelated to this change).

## Files Changed

| File | Change |
|------|--------|
| `src/features/augment.py` | +184 lines: 7 new methods, updated `random_translation`, 6 new calls in `__call__` |
| `tests/test_features.py` | +141 lines: `TestNewAugmentations` class with 21 tests, added `LandmarkAugmentor` import |

## Self-Review Findings

1. **`random_translation` default changed**: `max_shift` increased from 0.05 to 0.1 to match the spec's [-0.1, 0.1] range. This is a breaking change for any code relying on the old default — acceptable since this aligns with the KD-MSLRT research specification.

2. **`frame_permutation` segment detection**: The algorithm detects non-relevant segments by looking at consecutive-frame variation. It correctly handles edge cases (entire sequence low-var, entire sequence high-var, short sequences).

3. **`speed_variation` uses nearest-neighbor interpolation**: Frames are sampled via `int(np.linspace(...))`. For production quality, linear interpolation across the 168-dim vector would be smoother, but nearest-neighbor preserves discrete landmark semantics better and is consistent with the frame-drop augmentation's approach.

4. **No performance concerns**: All augmentations are O(T × D) or better. The `frame_permutation` segment scan is O(T × D). The `spatial_rotation` matrix operations are constant-time per frame.

## Issues or Concerns

- None identified. All acceptance criteria met.
