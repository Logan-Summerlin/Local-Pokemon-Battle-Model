# Experiment AR-009: lr_scaled_ls05

## Hypothesis
AR-008 showed 3.5pp train-val gap at convergence. Label smoothing 0.05 should reduce overfitting and improve generalization, potentially closing the 1.11pp gap to anchor.

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "label_smoothing": 0.05,
  "lr": 0.0004,
  "num_workers": 4,
  "warmup_steps": 75
}
```

## Expected Impact
Top-1 62.5-64%, reduced overfitting

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
