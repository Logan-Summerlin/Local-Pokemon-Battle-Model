#!/bin/bash
# Run a 2-stage curriculum experiment with the champion w5 config.
# Usage: bash scripts/run_curriculum_experiment.sh <name_suffix> <extra_overrides...>
#
# Example:
#   bash scripts/run_curriculum_experiment.sh value_head no_value_head=false
#   bash scripts/run_curriculum_experiment.sh deeper_policy policy_head_layers=2
#   bash scripts/run_curriculum_experiment.sh action_attn action_self_attention=true

set -e

NAME_SUFFIX="$1"
shift
EXTRA_OVERRIDES=("$@")

# Base config shared by all curriculum experiments
BASE_OVERRIDES=(
    num_layers=5
    hidden_dim=256
    num_heads=4
    ffn_multiplier=3
    batch_size=1024
    lr=4e-4
    warmup_steps=300
    patience=20
    max_window=5
    split_head=true
    shuffle_moves=true
    move_identity=true
    no_value_head=true
    amp=bf16
)

echo "================================================================"
echo "CURRICULUM EXPERIMENT: ${NAME_SUFFIX}"
echo "Extra overrides: ${EXTRA_OVERRIDES[*]}"
echo "================================================================"

# ── Stage 1 ──
echo ""
echo ">>> STAGE 1: Foundation (1100-1300 Elo)"
python Autoresearch/run_experiment.py \
    --name "t2_curr_w5_${NAME_SUFFIX}_s1" \
    --parent "AR-033" \
    --tier 2 \
    --budget-epochs 20 \
    --hypothesis "Curriculum w5 Stage 1 with ${NAME_SUFFIX}" \
    --config-override \
        "${BASE_OVERRIDES[@]}" \
        epochs=20 \
        cosine_epochs=40 \
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
    --name "t2_curr_w5_${NAME_SUFFIX}_s2" \
    --parent "$(python -c "import json; print(json.load(open('Autoresearch/experiment_registry.json'))[-1]['experiment_id'])")" \
    --tier 2 \
    --budget-epochs 25 \
    --hypothesis "Curriculum w5 Stage 2 with ${NAME_SUFFIX}" \
    --config-override \
        "${BASE_OVERRIDES[@]}" \
        epochs=25 \
        cosine_epochs=50 \
        warmup_steps=100 \
        patience=10 \
        battle_manifest=data/curriculum/stage2.json \
        "resume_from=${STAGE1_CKPT}" \
        "${EXTRA_OVERRIDES[@]}"

echo ""
echo "================================================================"
echo "CURRICULUM EXPERIMENT COMPLETE: ${NAME_SUFFIX}"
echo "================================================================"
