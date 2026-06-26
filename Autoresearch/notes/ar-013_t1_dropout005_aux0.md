# Experiment AR-013: t1_dropout005_aux0

## Hypothesis
Both dropout 0.05 (+0.48pp) and aux_weight 0 (+0.33pp) showed positive signal. Combining may stack.

## Change Made
Parent: AR-010

```json
{
  "aux_weight": 0.0,
  "dropout": 0.05
}
```

## Expected Impact
Top-1 +0.5-1pp from stacking two positive signals

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
