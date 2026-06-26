# FIX 002: Checkpoints Do Not Persist Policy-Head Config Flags

## Priority: HIGH — Fix before any standalone evaluation, online play, or checkpoint reuse

## Problem

`save_checkpoint()` in `scripts/train_phase4.py:749-780` writes a `config` dict that
omits every policy-head architecture flag introduced after the original P8-Lean design:

- `use_split_head`
- `use_candidate_head`
- `move_identity_candidates`
- `policy_head_layers`
- `action_self_attention`

It also omits the loss-tuning fields `switch_weight` and `label_smoothing` (not
load-breaking, but needed for reproducibility) and `max_seq_len`.

Any loader that reconstructs the model from `ckpt["config"]` — including
`Autoresearch/eval_harness.py:load_checkpoint` (lines 65-101) and any future
`ModelBot` loader — builds the default pooled-MLP policy head and fails with a
`state_dict` mismatch (or worse, silently mis-evaluates if `strict=False` is ever used).

**Consequence:** the champion AR-020 checkpoint (split head + action self-attention +
move-identity candidates) cannot be loaded outside its own training run. All registry
metrics for recent experiments come from `train_phase4.py`'s in-run test evaluation, not
the standalone harness. Online evaluation (Phase C) is blocked until this is fixed.

## Root Cause Trace

| Stage | File | Status |
|-------|------|--------|
| 1. Config defines head flags | `src/models/battle_transformer.py:90-95` | Yes |
| 2. Model builds head from flags | `src/models/battle_transformer.py:1060-1064` | Yes |
| 3. save_checkpoint persists flags | `scripts/train_phase4.py:757-779` | **No** |
| 4. eval_harness reconstructs from config | `Autoresearch/eval_harness.py:70-92` | **No** |

## Fix

1. In `scripts/train_phase4.py:save_checkpoint`, add to the `config` dict:

```python
"use_candidate_head": config.use_candidate_head,
"use_split_head": config.use_split_head,
"move_identity_candidates": config.move_identity_candidates,
"policy_head_layers": config.policy_head_layers,
"action_self_attention": config.action_self_attention,
"switch_weight": config.switch_weight,
"label_smoothing": config.label_smoothing,
"max_seq_len": config.max_seq_len,
```

2. In `Autoresearch/eval_harness.py:load_checkpoint`, pass the same keys through to
   `TransformerConfig`, using `.get(key, <dataclass default>)` so that pre-fix
   checkpoints (P8-Lean, early AR runs) still load correctly.

3. **Re-save the champion:** load AR-020's `best_model.pt` weights with the known config
   (from the experiment registry entry) and re-save with the complete config dict.
   Keep the original file until the re-saved copy is verified.

## Files to Modify
- `scripts/train_phase4.py` — `save_checkpoint()`
- `Autoresearch/eval_harness.py` — `load_checkpoint()`
- Any future loader (e.g., `src/bots/model_bot.py` once ported) must read these flags.

## Verification
`eval_harness.py` loads the re-saved AR-020 checkpoint without state_dict errors and
reproduces the registry metrics (67.79% top-1 / 93.42% top-3 within rounding) on the
test split. Also verify a pre-fix checkpoint (P8-Lean) still loads via the `.get`
defaults.
