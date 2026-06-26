# Experiment AR-008: batch256_lr_scaled

## Hypothesis
Previous batch_size=256 experiments may converge worse due to 4x fewer gradient updates with same LR. Linear scaling rule: multiply LR by batch_size ratio (256/64=4x) and reduce warmup proportionally. This should recover the anchor's convergence behavior while being 4x faster per epoch.

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "lr": 0.0004,
  "num_workers": 4,
  "warmup_steps": 75
}
```

## Expected Impact
Top-1 ~63% (matching anchor)

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
