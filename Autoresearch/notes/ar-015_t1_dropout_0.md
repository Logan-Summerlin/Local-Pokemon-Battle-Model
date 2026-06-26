# Experiment AR-015: t1_dropout_0

## Hypothesis
Maximum model capacity. At Tier 1 scale with 25K data the model isn't really overfitting, so dropout may just hurt.

## Change Made
Parent: AR-010

```json
{
  "dropout": 0.0
}
```

## Expected Impact
Top-1 +0.5-1pp if dropout is hurting at this scale

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
