# Experiment AR-030: t2_5L320d_wide_scaleup

## Hypothesis
Width scale-up 256→320 with proven 5L/FFN3x config. Both move_emb=48 and FFN 4x failed, so keeping defaults and testing pure width increase.

## Change Made
Parent: AR-027

```json
{
  "hidden_dim": 320
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
