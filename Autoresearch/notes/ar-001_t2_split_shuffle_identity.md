# Experiment AR-001: t2_split_shuffle_identity (baseline anchor)

First move-identity experiment and the frozen baseline anchor for all comparisons;
root of the registry (no parent). All later experiments are judged against this.

## Hypothesis
Move shuffle prevents slot memorization while move-identity candidates give the model a way to identify actual moves. Together they force the model to learn 'use Earthquake' instead of 'click slot 1'.

## Change Made
Parent: none (root / baseline anchor)

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
