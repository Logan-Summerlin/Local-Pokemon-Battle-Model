# AutoResearch Harness

This directory contains the experiment-management layer for the Phase 4 training pipeline.

## Main files

- `run_experiment.py` — register-first launcher for bounded experiments.
- `eval_harness.py` — checkpoint evaluation helper.
- `leaderboard.py` — experiment ranking.
- `experiment_registry.json` — machine-readable run registry.
- `configs/anchor.yaml` — reference experiment config.
- `notes/000_anchor.md` — baseline note.

## Standard loop

1. register or update an experiment,
2. launch `scripts/train_phase4.py` through `run_experiment.py`,
3. evaluate the resulting checkpoint,
4. record the outcome in the registry and notes.

## Example

```bash
python Autoresearch/run_experiment.py \
  --name window5 \
  --parent anchor \
  --tier 1 \
  --hypothesis "Longer battle context improves switch prediction." \
  --config-override max_window=5 \
  --budget-epochs 10

python Autoresearch/eval_harness.py \
  --checkpoint checkpoints/phase4_gen3_p8_lean/seed_42/best_model.pt \
  --data-dir data/processed

python Autoresearch/leaderboard.py
```
