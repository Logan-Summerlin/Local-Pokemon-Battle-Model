#!/bin/bash
# Local (GTX 1650, 4 GB) variant of the 2-stage champion-class curriculum.
#
# This mirrors scripts/run_curriculum_experiment.sh (the A40 recipe) but retunes
# every memory/precision knob for a single NVIDIA GTX 1650:
#   * batch_size 64 x grad_accum 16  == effective optimizer batch 1024 (no OOM)
#   * amp=fp16  (Turing has 2x packed FP16 on its CUDA cores but NO bf16 hardware,
#                so bf16 autocast is a pessimisation here — fp16 is faster + lighter)
#   * reduced epochs and an explicit, generous --budget-minutes per stage, because
#     the Tier-2 wall-clock cap (120 min) assumes an A40, not a 1650.
#
# Prereq: curriculum manifests must exist (data/curriculum/stage{1,2}.json).
#   python scripts/prepare_curriculum_data.py        # downloads + builds manifests
# If you do NOT want to rebuild the Elo-stratified manifests, use the single-stage
# out-of-the-box path instead (trains on the committed 25K battles, no download):
#   python Autoresearch/run_experiment.py --name local_pretrain --parent anchor \
#       --tier 2 --profile gtx1650 --budget-minutes 360 \
#       --config-override epochs=12 cosine_epochs=24
#
# Usage: bash scripts/run_curriculum_experiment_local.sh <name_suffix> <extra_overrides...>
#   Example: bash scripts/run_curriculum_experiment_local.sh action_attn action_self_attention=true

set -e

NAME_SUFFIX="$1"
shift
EXTRA_OVERRIDES=("$@")

# Base config shared by all local curriculum experiments (GTX 1650 safe).
BASE_OVERRIDES=(
    num_layers=5
    hidden_dim=256
    num_heads=4
    ffn_multiplier=3
    batch_size=64
    grad_accum=16          # effective batch 1024
    lr=4e-4
    warmup_steps=300
    patience=10
    max_window=5
    split_head=true
    shuffle_moves=true
    move_identity=true
    no_value_head=true
    amp=fp16               # NOT bf16 on Turing
    num_workers=6
)

# Per-stage wall-clock budgets (minutes). Tune to your measured throughput; together
# with the post-training pass these should stay under the local ~8 h target.
STAGE1_BUDGET=210
STAGE2_BUDGET=210

echo "================================================================"
echo "LOCAL CURRICULUM EXPERIMENT (GTX 1650): ${NAME_SUFFIX}"
echo "Extra overrides: ${EXTRA_OVERRIDES[*]}"
echo "================================================================"

# ── Stage 1 ──
echo ""
echo ">>> STAGE 1: Foundation (1100-1300 Elo)"
python Autoresearch/run_experiment.py \
    --name "local_curr_w5_${NAME_SUFFIX}_s1" \
    --parent "anchor" \
    --tier 2 \
    --budget-epochs 12 \
    --budget-minutes "${STAGE1_BUDGET}" \
    --hypothesis "Local GTX1650 curriculum w5 Stage 1 with ${NAME_SUFFIX}" \
    --config-override \
        "${BASE_OVERRIDES[@]}" \
        epochs=12 \
        cosine_epochs=24 \
        battle_manifest=data/curriculum/stage1.json \
        resume_from=none \
        "${EXTRA_OVERRIDES[@]}"

# Find the checkpoint dir from the latest experiment
STAGE1_CKPT=$(python -c "
import json
reg = json.load(open('Autoresearch/experiment_registry.json'))
print(reg[-1].get('checkpoint_path', reg[-1].get('checkpoint_dir', '')))
")

echo ""
echo ">>> Stage 1 checkpoint: ${STAGE1_CKPT}"
echo ""

# ── Stage 2 ──
echo ">>> STAGE 2: Specialization (1300+ Elo)"
python Autoresearch/run_experiment.py \
    --name "local_curr_w5_${NAME_SUFFIX}_s2" \
    --parent "$(python -c "import json; print(json.load(open('Autoresearch/experiment_registry.json'))[-1]['experiment_id'])")" \
    --tier 2 \
    --budget-epochs 12 \
    --budget-minutes "${STAGE2_BUDGET}" \
    --hypothesis "Local GTX1650 curriculum w5 Stage 2 with ${NAME_SUFFIX}" \
    --config-override \
        "${BASE_OVERRIDES[@]}" \
        epochs=12 \
        cosine_epochs=24 \
        warmup_steps=100 \
        battle_manifest=data/curriculum/stage2.json \
        "resume_from=${STAGE1_CKPT}" \
        "${EXTRA_OVERRIDES[@]}"

echo ""
echo "================================================================"
echo "LOCAL CURRICULUM EXPERIMENT COMPLETE: ${NAME_SUFFIX}"
echo "================================================================"
