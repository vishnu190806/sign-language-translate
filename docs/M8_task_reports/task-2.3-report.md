# Task 2.3: Knowledge Distillation

## What Was Implemented

Created `src/train/distiller.py` with a `Distiller` class that implements knowledge distillation from a larger teacher model to a lightweight student model.

### Components

1. **`Distiller` class** — orchestrates the distillation pipeline
   - Constructor accepts `teacher`, `student` (both `HTMAClassifier`), `temperature` (default 4.0), `alpha` (default 0.7), and optional `feature_weight`
   - Teacher is frozen (`requires_grad=False`, eval mode)
   - Student is trainable

2. **KD Loss** — KL divergence between soft targets
   - `soft_teacher = softmax(teacher_logits / T)`
   - `soft_student = log_softmax(student_logits / T)`
   - `kd_loss = KL(soft_student || soft_teacher) * T^2`

3. **Combined Loss** — weighted sum of KD and CE
   - `loss = alpha * kd_loss + (1 - alpha) * ce_loss`
   - Optional feature distillation via MSE on embedding vectors when `feature_weight > 0`

4. **`train_epoch(dataloader, optimizer, device)`** — full training loop
   - Teacher forward pass under `torch.no_grad()`
   - Student forward pass, combined loss, backward, grad clipping

5. **`validate(dataloader, device)`** — evaluation loop
   - Student-only forward pass, returns loss and accuracy

6. **`export_student(path)`** — saves student state dict and JSON config

### Helper Module

- **`FeatureDistillLoss`** — MSE between student/teacher feature vectors with optional linear projection when embedding dims differ

## Files Changed

| File | Action |
|------|--------|
| `src/train/distiller.py` | Created |
| `tests/test_distiller.py` | Created |

## Tests

Created 15 unit tests across 7 test classes:

- **`TestTeacherFrozen`** (3) — verifies no grad, eval mode, no grad_fn on output
- **`TestStudentTrainable`** (2) — verifies requires_grad=True and train mode
- **`TestKDLoss`** (2) — verifies loss computes and scales with temperature
- **`TestCombinedLoss`** (3) — verifies summation, alpha=0 reduces to CE, alpha=1 reduces to KD
- **`TestTrainEpoch`** (2) — verifies metrics returned and weights updated
- **`TestValidate`** (1) — verifies metrics returned with valid ranges
- **`TestExportStudent`** (2) — verifies file creation and config content

**Result: 15/15 passed**

Full suite: 190 passed, 16 pre-existing failures (missing dataset file + missing `scripts.category_analysis` module — unrelated).

## Self-Review Findings

- No comments added (per AGENTS.md)
- Type hints used throughout
- Follows existing patterns from `trainer.py` (same import style, same metric dict convention)
- No deviations from the task spec

## Issues or Concerns

None.
