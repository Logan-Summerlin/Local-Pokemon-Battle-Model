# Experiment AR-000: Anchor (P8-Lean 50K)

## Description

Frozen anchor checkpoint for all AutoResearch comparisons. Trained on GTX 1650 with severe compute constraints.

## Configuration

- **Architecture**: 3L / 224d / 4H / FFN 3x (~1.72M parameters)
- **Context window**: 2 turns
- **Data**: 50K Gen 3 OU battles (Metamon dataset), 694K train examples
- **Training**: batch_size=64, lr=1e-4, warmup 300 steps, cosine annealing
- **Heads**: Policy + Auxiliary (item/speed/role), no Value head
- **Hardware**: GTX 1650 (4GB VRAM)

## Results

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 63.21% |
| Top-3 accuracy | 89.27% |
| Policy NLL | 1.0148 |
| Move accuracy | ~72–75% |
| Switch accuracy | ~37–48% |
| Aux item accuracy | ~unknown (needs eval_harness run) |
| Aux speed accuracy | 0% (broken) |
| Aux role accuracy | 0% (broken) |
| Throughput | ~1,520 ex/sec |
| Wall time | 212 min |

## Known Issues

1. **Auxiliary speed and role heads show 0% accuracy across all epochs.** This is a bug — either in label construction (`src/data/auxiliary_labels.py`), loss masking in `compute_total_loss()`, or gradient flow. Must be debugged in AR-1 before any aux-related experiments.

2. **Switch prediction is dramatically worse than move prediction** (37–48% vs 72–75%). Switches are ~35% of all actions. Closing half this gap would add ~5 points to overall Top-1.

3. **Calibration shows systematic overconfidence** in the 0.4–0.8 confidence range.

4. **Only 50K of >100K available battles used.** Training on the full dataset is an obvious early experiment.

## Decision

PROMOTE — this is the frozen anchor, not a candidate.

## Next Steps

1. Run eval_harness.py on this checkpoint to get exact move/switch/aux/calibration numbers
2. Debug auxiliary speed/role heads (AR-1)
3. Window size sweep: 2 → 3 → 5 → 8 (AR3-02)
4. Full dataset training (AR3-01)
5. Hypothesize next experiments, and run them
