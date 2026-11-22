# Task 3.2 — Per-Signer Normalization

## Status: DONE

## What Was Implemented

Three new classes added to `src/features/normalize.py`:

1. **SignerStatistics** — Computes per-signer mean/std across all frames and normalizes features by subtracting mean and dividing by std (with small-std clamping to 1.0).

2. **SignerEstimator** — Estimates signer identity from the first few frames (up to 5) by computing MSE between the query's normalized features and each signer's statistics. Returns the signer ID with lowest MSE.

3. **PerSignerNormalizer** — Combines normalization with signer estimation. When `use_per_signer=True` and signer statistics are provided, it estimates the signer and applies per-signer normalization. Otherwise falls back to identity (no change).

Integration: `build_frame_features` now accepts optional `per_signer_normalizer`, `signer_stats`, and `signer_id` parameters. When `per_signer_normalizer` is provided with `use_per_signer=True`, per-signer normalization is applied to the assembled feature vector. All new parameters default to `None`, preserving full backward compatibility.

## Files Changed

- `src/features/normalize.py` — Added `SignerStatistics`, `SignerEstimator`, `PerSignerNormalizer` classes; extended `build_frame_features` with optional per-signer normalization parameters.
- `tests/test_features.py` — Added `TestSignerStatistics` (6 tests), `TestSignerEstimator` (2 tests), `TestPerSignerNormalizer` (5 tests), `TestBuildFrameFeaturesBackwardCompatibility` (2 tests).

## Test Results

- **44/44 feature tests pass** (15 new + 29 existing)
- Full suite: 218 passed, 16 pre-existing failures (missing dataset files and `scripts.category_analysis` module — unrelated)

## Self-Review Findings

- Initial implementation used cosine similarity for signer estimation, which failed on 2D arrays. Switched to MSE-based matching which works correctly with 1D averaged feature vectors.
- `compute_statistics` initially used `np.stack` which preserved the temporal dimension. Fixed to use `np.concatenate` to properly flatten across time, producing (168,) mean/std vectors.
- No comments added per AGENTS.md convention.

## Acceptance Criteria

- [x] SignerStatistics class implemented
- [x] SignerEstimator class implemented
- [x] PerSignerNormalizer class implemented
- [x] Integration with existing code works
- [x] Backward compatibility maintained
- [x] Unit tests pass
- [x] Code follows existing patterns
- [x] No comments added
