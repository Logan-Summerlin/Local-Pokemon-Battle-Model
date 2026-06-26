# Experiment AR-005: switch_weight_2x

## Hypothesis
Switch accuracy is 42.75% vs 74.17% for moves. Upweighting switch loss 2x forces the model to allocate more capacity to distinguishing correct switch targets.

## Change Made
Parent: AR-000

```json
{
  "switch_weight": 2.0,
  "batch_size": 256,
  "amp": "bf16",
  "num_workers": 4
}
```

## Expected Impact
Top-1 +1-2pp, switch accuracy +5-10pp

## Results
| Metric | Value |
|--------|-------|
| Top-1 | 61.85% |
| Top-3 | 89.60% |
| Move acc | 64.9% |
| Switch acc | 56.1% |

## Delta From Parent
| Metric | Delta |
|--------|-------|
| Top-1 | -1.36pp |
| Top-3 | +0.33pp |
| Move acc | -9.3pp |
| Switch acc | +13.3pp |

## Analysis

Switch_weight 2x shows a clear trade-off:
- Switch accuracy massively improved (+13.3pp) but move accuracy degraded (-9.3pp)
- 2x weight is too aggressive — overcorrects by sacrificing moves
- Sweet spot should be 1.3-1.5x to gain switch accuracy with less move cost
- Key insight: switch prediction IS learnable with loss weighting

## Decision
- [x] KILL (but learnings inform next experiment: try 1.5x)
- [ ] RETRY
- [ ] PROMOTE
