# Experiment AR-003: full_100k

## Hypothesis
More training data (80K vs 50K battles) should improve generalization, especially for rarer switch patterns

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "epochs": 10,
  "num_battles": 100000,
  "num_workers": 4
}
```

## Expected Impact
Top-1 +0.5-2pp from more diverse training data

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
