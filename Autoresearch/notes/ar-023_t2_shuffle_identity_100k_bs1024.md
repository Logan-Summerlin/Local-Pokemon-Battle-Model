# Experiment AR-023: t2_shuffle_identity_100k_bs1024

## Hypothesis
More compute (100K battles, 40 epochs) with larger batch (1024) and sqrt-scaled LR (8e-4) gives the shuffle+identity model enough training to learn move-specific patterns

## Change Made
Parent: AR-022

```json
{
  "batch_size": 1024,
  "epochs": 40,
  "lr": 0.0008,
  "num_battles": 100000,
  "warmup_steps": 150
}
```

## Expected Impact
Close the gap between the move-identity model (55.09%) and the slot-based champion (67.41%) while maintaining correct move-identity generalization

## Results
```json
{}
```

## Delta From Parent
```json
{}
```

## Analysis
TODO

## Decision
- [ ] KILL
- [ ] RETRY
- [ ] PROMOTE
