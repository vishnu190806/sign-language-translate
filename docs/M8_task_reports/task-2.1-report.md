# Task 2.1 — Test-Time Adaptation (SAME Framework)

## Status: DONE

## What Was Implemented

### `src/models/same_adapter.py` (115 lines)

**MoELayer** — Lightweight Mixture-of-Experts module inserted after each HTMA block:
- N LoRA-based experts: `Linear(D→bottleneck) → ReLU → Linear(bottleneck→D)`
- Learnable alpha parameter balancing expert output vs pass-through (residual)
- Signer-aware gating: `Linear(6→num_experts)` with softmax weights

**SAMEAdapter** — Wraps `HTMAClassifier` with test-time adaptation:
- Freezes all source model parameters
- Inserts one `MoELayer` after each `HTMABlock`
- `_extract_signer_features()`: computes 6-dim signer representation from 168-dim input (mean/std of left hand, right hand, pose regions)
- `forward()`: replicates `HTMAClassifier.forward()` with MoE insertions after each block
- `adapt()`: unsupervised TTA loop with three objectives:
  1. **Entropy minimization**: `-mean(p * log(p))`
  2. **Minimum Class Confusion (MCC)**: `-sum(sqrt(p * q))` where `q = p²/Σp²`
  3. **Pseudo-label KL divergence**: `KL(p_source || p_adapted)` to prevent drift
- Adam optimizer at `lr=1e-4`, only updates adapter parameters

### `tests/test_same_adapter.py` (12 tests)

| Test | Description |
|------|-------------|
| `test_forward_shape` | MoE layer preserves (B, T, D) shape |
| `test_expert_count` | Correct number of experts created |
| `test_inserts_moe_after_each_block` | One MoE per HTMA block |
| `test_original_params_frozen` | Source model `requires_grad=False` |
| `test_adapter_params_trainable` | Adapter has trainable parameters |
| `test_forward_shape` | Full adapter outputs (B, num_classes) |
| `test_forward_with_mask` | Mask support works |
| `test_adapt_runs` | Adaptation loop completes without NaN |
| `test_adapt_does_not_update_source_params` | Source weights unchanged after adapt |
| `test_signer_features_shape` | Signer features are (B, 6) |
| `test_signer_features_values` | Known input produces expected stats |
| `test_unsupervised_losses` | All three losses compute valid values |

## Test Results

```
tests/test_same_adapter.py — 12 passed
Full suite — 159 passed, 16 failed (pre-existing, unrelated)
```

Pre-existing failures: 10 from `test_dataset.py` (missing `isl_label_map.json`), 6 from `test_m7.py` (missing `scripts.category_analysis` module).

## Files Changed

| File | Action |
|------|--------|
| `src/models/same_adapter.py` | Created |
| `tests/test_same_adapter.py` | Created |

## Self-Review Findings

- No comments added (per AGENTS.md)
- Type hints used throughout
- Code follows existing patterns from `htma.py` and `common.py`
- All tensor operations are PyTorch-native
- Gradient flow is correct: frozen source parameters still participate in computation graph, allowing adapter parameters to receive gradients via chain rule

## Concerns

- **No integration with training pipeline yet**: The adapter is standalone. A future task should integrate it into the inference pipeline and add a TTA evaluation benchmark.
- **Signer feature extraction is simple**: Mean/std of body regions is a lightweight proxy. More sophisticated signer embeddings (e.g., from a pretrained signer encoder) could improve gating quality.
