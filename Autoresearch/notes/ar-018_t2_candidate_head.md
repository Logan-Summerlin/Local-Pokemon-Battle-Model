# Experiment AR-018: t2_candidate_head

## Hypothesis
Current pooled-MLP head treats switches as abstract slot IDs. Candidate head lets model reason about actual switch targets (specific bench pokemon) via cross-attention over encoder memory. Should dramatically improve switch accuracy (currently 37-48%).

## Change Made
Parent: AR-000

```json
{
  "amp": "bf16",
  "batch_size": 256,
  "candidate_head": true,
  "lr": 0.0004,
  "num_workers": 4,
  "warmup_steps": 75
}
```

## Expected Impact
Switch accuracy +10-20pp, Top-1 +1-3pp

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
