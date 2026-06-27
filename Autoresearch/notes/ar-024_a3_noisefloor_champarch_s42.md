# Experiment AR-024: a3_noisefloor_champarch_s42

## Hypothesis
Establish local champion-arch noise floor (5.65M params, action-attn+2L policy head, 25K battles, fp16, single-stage) — seed 42 of 3.

## Change Made
Parent: AR-020

```json
{
  "amp": "fp16",
  "batch_size": 256,
  "battle_manifest": null,
  "cosine_epochs": 20,
  "epochs": 10,
  "grad_accum": 4,
  "num_battles": 25000,
  "prefetch_factor": 2
}
```

## Expected Impact
Documents mean+-sigma top-1/switch/ECE for the local champion recipe; baseline for all Phase B deltas.

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
