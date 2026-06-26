# Experiment AR-044: t2_curr_w5_action_attn_bs4096_s1

## Hypothesis
4x batch size (4096) with sqrt-scaled LR (8e-4) may improve generalization through larger effective gradient averaging while maintaining training speed

## Change Made
Parent: AR-041

```json
{
  "batch_size": 4096,
  "lr": 0.0008
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
