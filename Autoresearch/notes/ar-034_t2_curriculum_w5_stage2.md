# Experiment AR-034: t2_curriculum_w5_stage2

## Hypothesis
Curriculum Stage 2 with window=5. Stage 1 already beat the w2 champion at 59.50%, so Stage 2 should push well above 60%.

## Change Made
Parent: AR-033

```json
{
  "battle_manifest": "data/curriculum/stage2.json",
  "cosine_epochs": 50,
  "epochs": 25,
  "patience": 10,
  "resume_from": "checkpoints/autoresearch_ar-033_t2_curriculum_w5_stage1",
  "warmup_steps": 100
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
