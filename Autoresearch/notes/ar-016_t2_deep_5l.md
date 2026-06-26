# Experiment AR-016: t2_deep_5L

## Hypothesis
More depth adds representational capacity through more processing steps. Same width means less parameter growth than wider model.

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "lr": 0.0004,
  "num_layers": 5,
  "num_workers": 4,
  "warmup_steps": 75
}
```

## Expected Impact
Top-1 62-64%

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
