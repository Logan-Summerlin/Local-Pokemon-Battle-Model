# Experiment AR-002: window5_t2

## Hypothesis
More battle history gives the model richer context. AR-001 timed out at epoch 6 still improving. Need full training to converge.

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "max_window": 5,
  "num_workers": 4
}
```

## Expected Impact
Top-1 +1-3pp over anchor

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
