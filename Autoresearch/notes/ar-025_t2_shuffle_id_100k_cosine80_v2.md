# Experiment AR-025: t2_shuffle_id_100k_cosine80_v2

## Hypothesis
Cosine schedule targeting 80 epochs keeps LR at ~50% of peak at epoch 40, preventing the LR death that wasted 5 epochs in AR-023

## Change Made
Parent: AR-023

```json
{
  "cosine_epochs": 80
}
```

## Expected Impact
Improve over 58.33% by maintaining useful LR through all 40 epochs

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
