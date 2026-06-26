# Experiment AR-022: t2_split_shuffle_identity

## Hypothesis
Move shuffle prevents slot memorization while move-identity candidates give the model a way to identify actual moves. Together they force the model to learn 'use Earthquake' instead of 'click slot 1'.

## Change Made
Parent: AR-019

```json
{
  "move_identity": true,
  "shuffle_moves": true
}
```

## Expected Impact
Should learn slower initially but generalize better. Key test: does the model's move probability follow the move identity when slots are swapped?

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
