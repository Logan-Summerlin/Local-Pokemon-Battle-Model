# Experiment AR-001: window5

## Hypothesis
More battle history gives the model richer context for predicting the next action, especially for switches that depend on HP/status trends

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "epochs": 10,
  "max_window": 5,
  "num_workers": 4
}
```

## Expected Impact
Top-1 +1-3pp, switch accuracy improvement

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
