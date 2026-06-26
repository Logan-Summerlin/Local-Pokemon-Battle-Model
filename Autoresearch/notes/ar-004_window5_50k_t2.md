# Experiment AR-004: window5_50k_t2

## Hypothesis
Expanding context from 2 to 5 turns gives the model history of HP trends, status, and previous actions - critical for switch prediction (currently 42.75%). AR-001 showed improvement trajectory at epoch 6 before timeout.

## Change Made
Parent: AR-000

```json
{
  "max_window": 5,
  "batch_size": 256,
  "amp": "bf16",
  "num_workers": 4
}
```

## Expected Impact
Top-1 +1-3pp, switch accuracy +5-10pp

## Results
```json
{
  "test_top1_accuracy": 0.5510,
  "test_top3_accuracy": 0.8156,
  "test_loss": 1.1859,
  "move_accuracy": 0.742,
  "switch_accuracy": 0.194,
  "wall_time_min": 82.14
}
```

## Delta From Parent
```json
{
  "test_top1_accuracy": -0.0811,
  "test_top3_accuracy": -0.0771,
  "move_accuracy": +0.000,
  "switch_accuracy": -0.234
}
```

## Analysis

Window 5 is a clear regression. Key findings:

1. **Move accuracy unchanged (74.2% vs 74.2%)** — the model learned moves equally well
2. **Switch accuracy catastrophically degraded (19.4% vs 42.8%)** — -23pp
3. The model converged at val_acc=0.551 after 30 epochs, never reaching anchor's 0.632
4. Training trajectory: epoch 1 val_acc=0.402, epoch 15 val_acc=0.539, epoch 30 val_acc=0.551
5. With 5x more tokens per example (70 vs 14), the 3L/224d model lacks capacity for effective temporal attention

### Root cause
The P8-Lean architecture (3L/224d) is too small to benefit from window 5. Additional temporal context introduces noise that overwhelms switch signals. Window expansion requires proportionally more model capacity and training epochs.

### For future experiments
- Do NOT retry window 5 without scaling up architecture to at least 4L/256d
- Focus on window 2 improvements first (loss engineering, data scaling)

## Decision
- [x] KILL
- [ ] RETRY
- [ ] PROMOTE
