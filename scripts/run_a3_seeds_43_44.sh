#!/usr/bin/env bash
# A3 noise floor: champion-arch local recipe, seeds 43 and 44 (seed 42 = AR-024).
# Run sequentially (one GPU). Launch only AFTER seed 42 (AR-024) has finished.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for SEED in 43 44; do
  echo "=== launching A3 seed ${SEED} at $(date -u) ==="
  python Autoresearch/run_experiment.py \
    --name "a3_noisefloor_champarch_s${SEED}" \
    --parent AR-020 --tier 2 --profile gtx1650 \
    --phase AR-A3 --family noise_floor --tags noise_floor a3 champarch "seed${SEED}" \
    --budget-minutes 450 \
    --config-override action_self_attention=True policy_head_layers=2 \
      battle_manifest=None resume_from=None epochs=10 cosine_epochs=20 \
      num_workers=4 prefetch_factor=2 "seed=${SEED}" \
    --hypothesis "Local champion-arch noise floor (5.65M params, 25K battles, fp16, single-stage) — seed ${SEED} of 3." \
    --expected-impact "Second/third seed for mean+-sigma top-1/switch/ECE noise floor."
  echo "=== seed ${SEED} finished at $(date -u) (exit $?) ==="
done
echo "=== A3 seeds 43+44 complete at $(date -u) ==="
