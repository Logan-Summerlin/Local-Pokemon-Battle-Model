# Experiment AR-006: t2_5L256d_lr4e4_warmup300

## Hypothesis
5L/256d with lower LR=4e-4 and warmup=300 to fix divergence seen in AR-005 at LR=8e-4

## Change Made
Parent: AR-004

```json
{
  "hidden_dim": 256,
  "lr": 0.0004,
  "num_layers": 5,
  "patience": 10,
  "warmup_steps": 300
}
```

## Expected Impact
TODO

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
