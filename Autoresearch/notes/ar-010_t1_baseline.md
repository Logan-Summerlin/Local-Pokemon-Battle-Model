# Experiment AR-010: t1_baseline

## Hypothesis
Establish fast baseline for Tier 1 comparisons. All Tier 1 experiments will be compared against this.

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "epochs": 5,
  "lr": 0.0004,
  "num_battles": 25000,
  "num_workers": 4,
  "warmup_steps": 75
}
```

## Expected Impact
Reference accuracy for Tier 1 budget

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
