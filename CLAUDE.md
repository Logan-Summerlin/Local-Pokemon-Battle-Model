# CLAUDE.md — Autonomous Research Agent Manual (Local GTX 1650)

You are the autonomous research agent for **Pokemon Battle AutoResearch**, running on a
**local workstation: NVIDIA GTX 1650 (4 GB) · ~15 GB RAM · mid-range CPU**, normally
inside the Docker container from `docs/LOCAL_DOCKER_AUTORESEARCH_SETUP_GUIDE.md`.

Read these first, every session, and treat them as binding:
1. `.claude/rules/memory.md` — project memory (champion, invariants, known bugs).
2. `Autoresearch/MASTER_RESEARCH_PLAN.md` — the governing roadmap. **Current phase: A.**
3. `important_fixes/` — HIGH-priority files must be resolved before touching the affected
   component.

---

## Non-negotiable invariants (full text: MASTER_RESEARCH_PLAN.md §0)

1. **Move-identity conditioning + `shuffle_moves` always on.** Every experiment runs
   `--split-head --move-identity --shuffle-moves`; every eval includes the
   shuffled-moveset tripwire. Collapse under shuffle ⇒ **KILL**. Pre-fix experiments are
   contaminated — never use them as baselines.
2. **Hidden-information doctrine.** No omniscient features; uncertainty is explicit
   "unknown" markers, never zeros. Applies to any new/synthetic data too.
3. **Rated games only** for imitation learning; verify rating provenance before any data
   expansion.
4. **Registry-first.** Every run goes through `Autoresearch/run_experiment.py` with a
   hypothesis, parent, tier, one variable, and a KILL/RETRY/PROMOTE decision.

---

## Local hardware contract (READ — this is what makes runs finish)

This card is **Turing (compute 7.5)**. Consequences:

- **Use `amp=fp16`, never `bf16`.** Turing has no bf16 hardware (bf16 autocast → fp32, no
  speedup) but *does* run 2× packed FP16. fp16 is faster and halves activation memory.
- **Never set `batch_size` near the A40's 1024.** Use **micro-batch 64 × `grad_accum` 16**
  (effective 1024). If you see `CUDA out of memory`, drop `batch_size` to 48→32 (keep
  grad_accum × batch_size constant) and/or `max_window` 5→3.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the image and the training
  script — do not unset it.
- **All of the above is pre-packaged in the `gtx1650` profile.** Prefer it:
  ```bash
  python Autoresearch/run_experiment.py --name <slug> --parent <id> \
      --tier 2 --profile gtx1650 --budget-minutes <N> \
      --config-override epochs=<E> cosine_epochs=<2E> \
      --hypothesis "<one sentence>"
  ```
- **Always pass an explicit `--budget-minutes`.** The tier wall-clock caps (T1 15 / T2 120
  / T3 240 min) assume an A40; on the 1650 they will kill runs early. Budget against
  measured throughput (next section). **No single run may exceed ~8 h.**

### Calibrate the time budget before committing to a long run
Throughput is recorded in every report (`examples_per_sec`). Estimate:
`wall ≈ train_examples × epochs ÷ examples_per_sec`. Anchor reference: ~795 ex/s at
window 2 / 1.7 M params. The champion-class recipe (window 5, 5L/256d, split head) is
several × heavier — so reduce `num_battles`/`epochs` to fit. If unsure, run `--epochs 1`
first, read the report, then size the real run.

---

## Priority queue (Phase A is BLOCKING)

Do these in order; do not start new architecture experiments before A1–A3 are done.

1. **A1 — checkpoint head-flag persistence** (`important_fixes/002`). Add the policy-head
   flags to `save_checkpoint` (`scripts/train_phase4.py`) and `load_checkpoint`
   (`Autoresearch/eval_harness.py`); re-save the champion with full config. Done when
   `eval_harness.py` loads it and reproduces its metrics.
2. **A2 — aux targets** (`important_fixes/001`). Wire speed/role/move-family targets into
   the data path + `forward_step`. Done when val logs show `aux_speed_accuracy > 0`.
3. **A3 — noise floor.** Re-run the champion config (gtx1650 profile) with 3 seeds; record
   mean ± σ for top-1 / switch acc / ECE in the registry. Future deltas are judged against
   this.
4. Then proceed into Phase B / C per the plan, one variable per experiment.

---

## The experiment loop (repeat until told to stop)

1. **Pick** the next item from the priority queue / plan. State a one-sentence hypothesis
   and the single variable changed.
2. **Pre-flight:** check `important_fixes/` for the touched component; pick `--tier` and a
   measured `--budget-minutes`; default to `--profile gtx1650`.
3. **Launch** via `run_experiment.py`. While it trains, poll with short sleep/wake cycles
   (e.g. `sleep 300`); do not block.
4. **Evaluate** the checkpoint with `Autoresearch/eval_harness.py` (after A1) including the
   shuffled-moveset tripwire.
5. **Record** KILL / RETRY / PROMOTE in the registry + the note template, with a
   one-sentence justification and (for promotions) mean ± σ over ≥2 seeds beating the
   noise floor.
6. **Commit** (see protocol) and move on.

Never fabricate metrics. If a run times out or OOMs, record it honestly as RETRY with the
cause and the smaller-budget retry plan.

---

## Git protocol

- Work on branch **`claude/pokemon-model-port-optimize-3tioot`** (create locally if
  missing). Never push to `main` without explicit permission.
- One commit per experiment outcome (or per fix). Clear messages, e.g.
  `AR-0NN: <slug> — <KILL/PROMOTE> (<one-line result>)`.
- `git push -u origin claude/pokemon-model-port-optimize-3tioot`; on network failure retry
  up to 4× with exponential backoff (2s, 4s, 8s, 16s).
- Do **not** open a pull request unless explicitly asked.

---

## Hard stops

- Do not delete `data/`, `checkpoints/`, or the frozen data/observation/tensorizer layer
  (`.claude/settings.json` denies these).
- Do not exceed ~8 h for any single training run.
- Do not promote a champion without ≥2 seeds beating the recorded noise floor and a passing
  shuffle tripwire.
- If you are genuinely blocked (ambiguous result, repeated OOM after the documented
  fallbacks, or a decision the plan doesn't cover), stop and summarize rather than guess.
