# Experiment AR-012: t1_aux_weight_0

## Hypothesis
Aux head predicts only items (speed/role broken). Zero aux weight focuses all capacity on policy loss.

## Change Made
Parent: AR-010

```json
{
  "aux_weight": 0.0
}
```

## Expected Impact
May improve or neutral — test if broken aux hurts

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
