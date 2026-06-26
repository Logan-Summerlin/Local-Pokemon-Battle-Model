# Experiment AR-021: t2_split_shuffle_moves

## Hypothesis
Per-battle move slot shuffling forces the model to learn move identities instead of slot positions, improving generalization and move accuracy

## Change Made
Parent: AR-019

```json
{
  "shuffle_moves": true
}
```

## Expected Impact
May initially lower Top-1 as slot shortcuts are removed, but should improve move selection quality and generalization. Expect improved accuracy once model learns actual move identity patterns.

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
