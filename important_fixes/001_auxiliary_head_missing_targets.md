# FIX 001: Auxiliary Head — Missing Speed, Role, and Move-Family Targets

## Priority: HIGH — Fix before running any aux-head experiments

## Problem

The `AuxiliaryHead` in `src/models/battle_transformer.py:640-702` predicts four targets:
- `item_logits` (25 item classes)
- `speed_logits` (5 speed buckets)
- `role_logits` (8 role archetypes)
- `move_family_logits` (10 move families)

But the training pipeline only provides **item_targets**. The other three heads
receive zero gradient and remain at random initialization — they are effectively
unsupervised. This is the root cause of the known "0% aux speed/role accuracy" bug.

## Root Cause Trace

The data path for auxiliary targets has four stages. The chain breaks at stage 2:

| Stage | File | Item | Speed | Role | Move Family |
|-------|------|------|-------|------|-------------|
| 1. Label extraction logic exists | `src/data/auxiliary_labels.py:428-486` | Yes | Yes | Yes | Yes |
| 2. Training script creates targets | `scripts/train_phase4.py:509-535` | **Yes** | No | No | No |
| 3. forward_step passes to loss | `scripts/train_phase4.py:342` | **Yes** | No | No | No |
| 4. Loss computed with gradient | `src/models/battle_transformer.py:947-1005` | **Yes** | No | No | No |

`add_auxiliary_labels()` in `train_phase4.py:509-535` manually extracts item indices
from `feat[5]` of the opponent team tensor. It never constructs speed, role, or
move-family targets. Meanwhile, `build_auxiliary_targets()` in `auxiliary_labels.py`
already returns all four target tensors — it just never gets called.

The loss function (`compute_auxiliary_loss`) silently skips any head whose target
key is missing from `aux_targets`, so no error is raised.

## Fix

### Option A: Call `build_auxiliary_targets()` from the training pipeline
The cleanest fix. `auxiliary_labels.py:build_auxiliary_targets()` already returns
a dict with all four target tensors. Integrate it into the data loading path so
that `speed_targets`, `role_targets`, and `move_family_targets` are added to each
sequence alongside `item_targets`.

### Option B: Extend `add_auxiliary_labels()` in train_phase4.py
Add speed/role/move-family extraction logic alongside the existing item extraction.
This duplicates logic that already exists in `auxiliary_labels.py`, so Option A
is preferred.

### In both cases, also update `forward_step()` (train_phase4.py:342):
```python
# Current (broken):
aux_targets = {"item_targets": batch["item_targets"]}

# Fixed:
aux_targets = {
    "item_targets": batch["item_targets"],
    "speed_targets": batch["speed_targets"],
    "role_targets": batch["role_targets"],
    "move_family_targets": batch["move_family_targets"],
}
```

## Files to Modify
- `scripts/train_phase4.py` — `add_auxiliary_labels()` + `forward_step()`
- `scripts/train_p8_lean.py` — if it has its own data path (check for duplication)

## Verification
After the fix, validation logs should show `aux_speed_accuracy > 0%` and
`aux_role_accuracy > 0%`. If they remain at 0%, check that the target tensors
contain valid (non -1) values for revealed opponent Pokemon.
