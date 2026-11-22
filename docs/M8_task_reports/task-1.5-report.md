# Task 1.5 — Augmentation Pipeline

## What I Implemented

Created `src/features/augment_pipeline.py` with:

- **`AugmentationPipeline`** class combining `LandmarkAugmentor` (13 augmentations) and `GeometryAugmentor` (6 augmentations) into a single unified interface
- **Configurable probabilities**: `set_prob()`, `enable()`, `disable()` methods for each augmentation
- **Statistics tracking**: records count, mean severity, and std severity per augmentation; `reset_statistics()` and `get_statistics()` methods
- **Mode-aware `__call__`**: returns input unchanged for `mode="val"` or `mode="test"`, applies augmentations for `mode="train"`
- **Visualization tools**: `visualize_augmentation()` (original vs augmented heatmap) and `visualize_statistics()` (frequency histogram + severity bar chart)
- **Factory function**: `create_augmentation_pipeline(config)` to build from a dict

## What I Tested and Test Results

Created `tests/test_augment_pipeline.py` with 22 tests across 7 test classes:

| Test Class | Tests | Status |
|---|---|---|
| TestPipelineShapePreservation | 2 | All pass |
| TestPipelineReproducibility | 2 | All pass |
| TestPipelineValMode | 4 | All pass |
| TestStatisticsTracking | 4 | All pass |
| TestConfiguration | 4 | All pass |
| TestFactoryFunction | 3 | All pass |
| TestVisualization | 3 | All pass |

**All 22 new tests pass.** Full suite: 147 pass, 16 pre-existing failures (missing dataset files and `scripts.category_analysis` module — not caused by this change).

## Files Changed

- `src/features/augment_pipeline.py` — new file (227 lines)
- `tests/test_augment_pipeline.py` — new file (190 lines)

## Self-Review Findings

- No comments added (per AGENTS.md)
- All type hints used
- Code follows existing patterns from `augment.py` and `augment_geo.py`
- Matplotlib is lazily imported inside visualization functions to avoid hard dependency

## Issues or Concerns

None.
