# AutoResearch — Project Memory

## Governing Plan
**Follow `Autoresearch/MASTER_RESEARCH_PLAN.md`.** Current phase: **A (Foundation repairs & rigor)**.
Phase A is blocking: complete A1 (checkpoint flags) and A2 (aux targets) and record the
noise floor (A3) before starting new experiments.

## Local Compute (current home)
Training runs on a single **NVIDIA GTX 1650 (4 GB, Turing) · ~15 GB RAM · mid-range CPU**,
usually in Docker (`docs/LOCAL_DOCKER_AUTORESEARCH_SETUP_GUIDE.md`). **Use `amp=fp16` (NOT
bf16 — Turing has no bf16 HW), micro-batch 256 × `grad_accum` 4, and an explicit
`--budget-minutes`; no run exceeds ~8 h.** The `gtx1650`/`local` profile in
`run_experiment.py`, `--mode local` in `train_phase4.py`, and
`run_curriculum_experiment_local.sh` encode this. Full details: MASTER_RESEARCH_PLAN.md §0.5.
The published champion (AR-020) was trained on an A40; its literal `batch_size=1024`/`bf16`
settings do not transfer to the 1650.

## Current Champion
- **AR-020** (`ar-020_t2_curr_w5_action_attn_s2`) — 67.79% Top-1, 93.42% Top-3, switch acc 60.51%
  - 5L/256d/4H, window=5, split_head + action_self_attention + move_identity + shuffle_moves,
    Elo-curriculum stage 2
- **Anchor (baseline)**: AR-001 (`t2_split_shuffle_identity`) — 55.09% Top-1, 85.56% Top-3
  - first move-identity experiment; root of the registry (no parent)

## Non-Negotiable Invariants (full text in MASTER_RESEARCH_PLAN.md §0)
1. **Move-identity conditioning + `shuffle_moves` always on.** The model chooses moves by
   identity (Solar Beam), never slot position (move2). Every eval includes the
   shuffled-moveset tripwire; collapse under shuffle = KILL. **All experiments predating
   the move-identity fix are contaminated — never use as baselines or evidence.**
2. **Hidden-information doctrine** — no omniscient features, explicit unknown markers;
   applies to self-play data too.
3. **Rated games only for imitation learning** — verify rating provenance before any
   dataset expansion (many undownloaded 1000–1500-bin Metamon battles are likely unrated).
4. **Registry-first** — hypothesis, parent, tier, one variable, KILL/RETRY/PROMOTE.

## Known Bugs
- **[FIXED — A1] Checkpoint head-flag persistence** (`important_fixes/002`).
  `save_checkpoint` now writes the policy-head/loss/seq flags and
  `eval_harness.load_checkpoint` reconstructs them via `.get(default)` (pre-fix
  checkpoints still load). Verified: split-head model round-trips with no
  state_dict error, reconstructs to 5,653,245 params (== AR-020 param_count).
  **Still open:** the actual AR-020 `best_model.pt` is NOT in this workspace
  (registry points to a separate `/workspace/Pokemon-Battle-Autoresearch`
  checkout), so it has not been re-saved/metric-reproduced here.
- **[FIXED — A2] Aux speed/role/move-family targets** (`important_fixes/001`).
  `add_auxiliary_labels(sequences, vocabs)` derives them from tensorized opponent
  features (base stats at feat idx 17–22 ÷255; move IDs decoded via moves vocab),
  threaded through dataset/collate/forward_step. Verified: val logs show
  aux speed≈0.39, role≈0.38 (was 0.0).
- Switch prediction 60.51% vs 71.86% for moves (biggest offline accuracy lever)
- Calibration: systematic overconfidence in 0.4–0.8 range

## Promotion Rules (summary)
- ≥2 seeds; delta must beat the recorded noise floor (Phase A3)
- Once online eval exists (Phase C): win-rate superiority over reigning champion,
  ≥400 paired battles, Wilson-CI significance; offline metrics become regression guards
- North star evolves: offline top-1 → gauntlet win rate → ladder GXE

## Important Fixes
Check `important_fixes/` before starting experiments. HIGH-priority files must be
resolved before running experiments that touch the affected component.
