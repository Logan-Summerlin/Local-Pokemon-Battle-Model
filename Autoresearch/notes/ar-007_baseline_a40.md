# Experiment AR-007: baseline_a40

## Hypothesis
All experiments used batch_size=256, anchor used 64. Need to verify if batch size change alone explains the ~1.5pp gap. This baseline establishes the A40 reference point.

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "num_workers": 4
}
```

## Expected Impact
Should match anchor 63.21% if hardware/bf16 is neutral

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
