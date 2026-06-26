# Experiment AR-006: switch_weight_1.5x

## Hypothesis
AR-005 showed 2x switch weight improved switch acc +13.3pp but degraded moves -9.3pp. 1.5x should find a better balance: moderate switch gains with less move cost, potentially beating anchor overall.

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "num_workers": 4,
  "switch_weight": 1.5
}
```

## Expected Impact
Top-1 +0.5-1.5pp, switch acc +5-8pp, move acc -3-5pp

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
