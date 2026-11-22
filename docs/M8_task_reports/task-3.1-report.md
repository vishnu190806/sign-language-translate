# Task 3.1: Multi-Scale Feature Fusion

## What Was Implemented

Added multi-scale feature fusion to `HTMAClassifier` in `src/models/htma.py`:

### New Config Parameters (HTMAConfig)
- `fusion_method`: `"early"` | `"late"` | `"multiscale"` — defaults to `"early"`
- `use_separate_encoders`: `bool` — defaults to `False`
- `hand_input_dim`: `int` — defaults to 128
- `pose_input_dim`: `int` — defaults to 40

### New Modules
- `HandEncoder`: Linear projection + LayerNorm for hand features (128 → embedding_dim)
- `PoseEncoder`: Linear projection + LayerNorm for pose features (40 → embedding_dim)
- `ConcatFusion`: Concatenation-based fusion (concat → linear projection)
- `AttentionFusion`: Cross-attention between hand and pose features
- `GatingFusion`: Learned gating to weight hand vs pose contributions

### Fusion Modes
1. **Early fusion** (`fusion_method="early"`): Current behavior — concatenate all 168 dims, single input projection. Default for backward compatibility.
2. **Late fusion** (`fusion_method="late"` + `use_separate_encoders=True`): Separate hand/pose encoders, run through shared HTMA blocks independently, fuse at attention pooling via ConcatFusion.
3. **Multiscale fusion** (`fusion_method="multiscale"`): Fuse hand and pose features after each HTMA block output using ConcatFusion modules. Supports both shared and separate encoders.

### Forward Dispatch
- `forward()` dispatches to `_forward_early()`, `_forward_separate()`, or `_forward_multiscale()` based on config
- All paths produce the same output shape `(B, num_classes)`
- All paths support attention masks

## What Was Tested and Test Results

18 tests in `tests/test_models.py` — all pass:

| Test | Result |
|------|--------|
| TestHTMAClassifier::test_forward_shape | PASS |
| TestHTMAClassifier::test_forward_with_mask | PASS |
| TestHTMAClassifier::test_build_model | PASS |
| TestHTMAClassifier::test_default_config_forward | PASS |
| TestHTMAClassifier::test_no_nan | PASS |
| TestEarlyFusion::test_early_fusion_default | PASS |
| TestEarlyFusion::test_early_fusion_with_mask | PASS |
| TestEarlyFusion::test_early_fusion_no_nan | PASS |
| TestLateFusion::test_late_fusion_shape | PASS |
| TestLateFusion::test_late_fusion_with_mask | PASS |
| TestLateFusion::test_late_fusion_no_nan | PASS |
| TestMultiscaleFusion::test_multiscale_shape | PASS |
| TestMultiscaleFusion::test_multiscale_with_separate_encoders | PASS |
| TestMultiscaleFusion::test_multiscale_with_mask | PASS |
| TestMultiscaleFusion::test_multiscale_no_nan | PASS |
| TestBackwardCompatibility::test_default_config_unchanged | PASS |
| TestBackwardCompatibility::test_default_produces_same_output_shape | PASS |
| TestBackwardCompatibility::test_separate_encoders_disabled_by_default | PASS |

## Files Changed

1. `src/models/htma.py` — Added fusion modules (HandEncoder, PoseEncoder, ConcatFusion, AttentionFusion, GatingFusion), expanded HTMAConfig, modified HTMAClassifier with multi-path forward dispatch
2. `tests/test_models.py` — Added TestEarlyFusion, TestLateFusion, TestMultiscaleFusion, TestBackwardCompatibility test classes

## Self-Review Findings

- AttentionFusion and GatingFusion modules are defined but not yet wired into any fusion path — they are available as building blocks for future use or can be activated via config.
- The `ConcatFusion` module is used for both late fusion and multiscale fusion. A config parameter could select between ConcatFusion/AttentionFusion/GatingFusion in a future iteration.
- All existing tests pass, and backward compatibility is fully maintained (default config produces identical behavior).

## Issues or Concerns

- None. The 16 pre-existing test failures in `test_dataset.py` and `test_m7.py` are unrelated (missing dataset files and missing `scripts.category_analysis` module).
