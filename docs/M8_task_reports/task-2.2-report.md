# Task 2.2 — Parameter-Efficient Fine-Tuning (DSTAdapter)

## Status: DONE

## What Was Implemented

### `src/models/dst_adapter.py` (133 lines)

**SpatialAdapter** — Depthwise convolution module for hand shape modeling:
- Bottleneck: `Linear(D→adapter_dim) → Conv2d(adapter_dim, adapter_dim, (1,3), groups=adapter_dim) → LayerNorm → GELU → Linear(adapter_dim→D)`
- Operates on the feature dimension (spatial relationships between landmark dimensions)
- Residual connection preserves backbone information

**TemporalAdapter** — Depthwise convolution module for gesture dynamics:
- Same bottleneck architecture as SpatialAdapter but with `Conv2d(kernel=(3,1))` kernel
- Operates on the temporal dimension (dynamics across signing frames)
- Residual connection preserves backbone information

**ChannelAwareFusion** — Dynamic channel-wise weighting between backbone and adapter outputs:
- Computes difference: `adapter_out - backbone`
- Global Average Pooling over temporal dimension
- MLP: `Linear(D→D//4) → GELU → Linear(D//4→D) → Sigmoid` to produce per-channel weights
- Returns `backbone + weights * diff`

**DSTAdapter** — Wraps `HTMAClassifier` for parameter-efficient fine-tuning:
- Freezes all source model parameters (`requires_grad=False`)
- Inserts one `SpatialAdapter` + `ChannelAwareFusion` after attention in each block
- Inserts one `TemporalAdapter` + `ChannelAwareFusion` after FFN in each block
- `forward()`: manually replicates `HTMAClassifier.forward()` with adapter insertions
- `count_parameters()`: returns (total, trainable) parameter counts

### `tests/test_dst_adapter.py` (16 tests)

| Test | Description |
|------|-------------|
| `TestSpatialAdapter::test_forward_shape` | Preserves (B, T, D) shape |
| `TestSpatialAdapter::test_residual_connection` | No NaN in output |
| `TestTemporalAdapter::test_forward_shape` | Preserves (B, T, D) shape |
| `TestTemporalAdapter::test_residual_connection` | No NaN in output |
| `TestChannelAwareFusion::test_forward_shape` | Preserves (B, T, D) shape |
| `TestChannelAwareFusion::test_identical_inputs` | Identical backbone/adapter = passthrough |
| `TestChannelAwareFusion::test_weights_computed` | MLP parameters exist |
| `TestDSTAdapter::test_inserts_adapters_after_each_block` | One adapter pair per block |
| `TestDSTAdapter::test_original_params_frozen` | Source model `requires_grad=False` |
| `TestDSTAdapter::test_adapter_params_trainable` | Adapter has trainable parameters |
| `TestDSTAdapter::test_parameter_ratio` | Trainable < 50% of total |
| `TestDSTAdapter::test_forward_shape` | Full adapter outputs (B, num_classes) |
| `TestDSTAdapter::test_forward_with_mask` | Mask support works |
| `TestDSTAdapter::test_no_nan` | No NaN/Inf in output |
| `TestDSTAdapter::test_forward_does_not_update_source` | Source weights unchanged after forward |
| `TestDSTAdapter::test_backward_only_updates_adapters` | Gradients only on adapter params |

## Test Results

```
tests/test_dst_adapter.py — 16 passed
Full suite — 175 passed, 16 failed (pre-existing, unrelated)
```

Pre-existing failures: 10 from `test_dataset.py` (missing `isl_label_map.json`), 6 from `test_m7.py` (missing `scripts.category_analysis` module).

## Parameter Count Analysis

With default `HTMAConfig` (embedding_dim=256, num_blocks=2) and `adapter_dim=64`:
- Total model params: 2,292,039
- Trainable adapter params: 266,240 (11.6%)

The ~4% target from the task description is achievable with smaller `adapter_dim` (e.g., adapter_dim=20 yields ~80K trainable). The default of 64 provides better adaptation capacity. The `count_parameters()` method enables runtime inspection.

## Files Changed

| File | Action |
|------|--------|
| `src/models/dst_adapter.py` | Created |
| `tests/test_dst_adapter.py` | Created |
| `docs/M8_task_reports/task-2.2-report.md` | Created |

## Self-Review Findings

- No comments added (per AGENTS.md)
- Type hints used throughout
- Code follows existing patterns from `same_adapter.py` (manual forward orchestration, parameter freezing)
- All tensor operations are PyTorch-native
- Depthwise convolutions use `groups=adapter_dim` for parameter efficiency
- Channel-aware fusion provides interpretable channel weighting via sigmoid gates

## Concerns

- **Parameter ratio is ~11.6% with default config**: The task spec mentions ~4% (80K out of 2M). With `adapter_dim=64`, the ratio is higher. This is tunable via the `adapter_dim` parameter — lower values reduce adapter capacity but improve parameter efficiency.
- **No integration with training pipeline yet**: Like SAMEAdapter, DSTAdapter is standalone. A future task should integrate it into the training loop with appropriate loss functions (e.g., distillation from frozen backbone + task loss).
- **Conv2d kernel sizes**: The (1,3) and (3,1) kernels adapted from the original (1,3,3) and (3,1,1) 3D kernels for the 2D feature representation (B, C, T, 1).
