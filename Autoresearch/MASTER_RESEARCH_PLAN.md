# MASTER RESEARCH PLAN — Pokemon-Battle-Autoresearch

**Status:** ACTIVE — this is the governing roadmap for the autoresearch agent.
**Current phase:** A (Foundation repairs & rigor)
**Mission:** Produce the strongest possible Gen 3 OU (ADV) battler. The final scoreboard is
**real Pokemon Showdown ladder play (Glicko/GXE)**, not offline accuracy.
**Compute assumption (current home):** a single **local workstation — NVIDIA GeForce
GTX 1650 (4 GB VRAM, Turing/compute 7.5) · ~15 GB system RAM · mid-range CPU**, normally
inside Docker (`docs/LOCAL_DOCKER_AUTORESEARCH_SETUP_GUIDE.md`), Node.js available (local
Showdown server is in scope). The earlier A40-class assumption (44 GB, 96 cores) still
applies to any cloud burst, but day-to-day work is local; see §0.5 for the hard
constraints this imposes.

---

## 0.5 Local Hardware Profile & Budget (binding while the home is the GTX 1650)

The frozen anchor (P8-Lean 50K) was trained on this exact card in ~3.5 h, so these numbers
are calibrated, not guessed. The published champion (AR-041) was trained on an A40 and its
literal settings (`batch_size=1024`, `amp=bf16`) **do not transfer**. Local rules:

- **Precision: `amp=fp16`, never `bf16`.** Turing has no bf16 hardware (bf16 autocast falls
  back to fp32 → zero speedup) but runs 2× packed FP16 on its CUDA cores. fp16 is both
  faster and halves activation memory.
- **Effective batch via accumulation.** Use micro-batch 256 × `grad_accum` 4 = effective
  1024 (the champion optimizer batch). The model is small (~4.6M params), so activations
  set the VRAM ceiling (~2.5–3.8 MB/example at window 5): batch 256 is ~1.5 GB, and the
  hard max is ~600–1200 depending on whether a display is attached. A literal 1024 is
  borderline (~3–4 GB) — avoid it for the safety margin, not because it always OOMs. On
  OOM, shrink `batch_size` 256→128→64 keeping `batch_size × grad_accum` constant, then
  drop `max_window` 5→3.
- **Allocator:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by the training
  script and Docker image) to avoid fragmentation OOMs.
- **Data/RAM:** keep the in-RAM windowed dataset to ~25K battles (~2–3 GB) so workers +
  OS fit in 15 GB.
- **Time:** **no single run exceeds ~8 h** (target: pretrain ≤ ~6 h + post-train ≤ ~1–2 h).
  Always pass an explicit `--budget-minutes`; the §1 tier wall-clock caps assume an A40.
  Budget against measured `examples_per_sec` (run `--epochs 1` first if unsure).
- **Canonical recipe:** the **`gtx1650`** profile (alias `local`) in
  `Autoresearch/run_experiment.py`, the `--mode local` preset in
  `scripts/train_phase4.py`, and `scripts/run_curriculum_experiment_local.sh` all encode
  the above. Prefer them over hand-rolled configs.

---

## 0. Non-Negotiable Invariants

These apply to every experiment in every phase. Violating any of them invalidates a run.

1. **Move-identity conditioning is mandatory.** The model must select actions by what the
   move *is* (e.g., Solar Beam), never by slot position (move2). The mid-program fix
   (`move_identity_candidates` + `shuffle_moves` augmentation, validated in AR-020/AR-022,
   part of champion AR-041) is permanent:
   - Every experiment runs with `shuffle_moves=true` and identity-conditioned candidate scoring.
   - Every evaluation includes a **shuffled-moveset generalization check**: re-evaluate with
     move slots randomly permuted. A model whose accuracy collapses under shuffle is
     position-cheating and is **KILLED** regardless of headline top-1.
   - **All experiments predating this fix are contaminated. Never use them as baselines,
     comparisons, or evidence.**
2. **Hidden-information doctrine** (inherited from Pokemon-Battle-Model, non-negotiable):
   never train on omniscient features unavailable at decision time; represent uncertainty
   with explicit "unknown" markers, not zeros; metagame priors are soft hints, not leaked
   truth. This applies to **all new data sources, including self-play recordings**.
3. **Rated games only for imitation learning.** Unrated battles (default/placeholder Elo)
   are excluded from all IL training data. Any dataset expansion must verify rating
   provenance first (see B-DATA-1).
4. **Registry-first.** Every run is registered via `Autoresearch/run_experiment.py` with a
   hypothesis, parent, and tier; outcomes recorded as KILL / RETRY / PROMOTE; one variable
   per experiment. Check `important_fixes/` before touching any affected component.

---

## 1. North-Star Metrics & Promotion Rules

The primary metric **evolves by phase**:

| Stage | North star | Role of other metrics |
|---|---|---|
| Phase A–B | Offline top-1 / top-3 / switch accuracy | ECE, aux accuracy as secondary |
| Phase C onward | **Gauntlet win rate** (fixed opponent suite, Wilson CI) | Offline metrics become regression guards only |
| Major versions | **Ladder GXE/Glicko** (≥400 ladder games) | Gauntlet as pre-flight check |

**Champion promotion rules:**
- ≥2 seeds confirming the result; delta must exceed the recorded noise floor (see A3).
- No offline regression beyond noise on held-out replay metrics.
- Once Phase C lands: win-rate superiority vs the reigning champion over ≥400 paired
  battles (alternating sides, shared team pool), non-overlapping Wilson 95% CIs or a
  paired test at p<0.05.
- Shuffled-moveset tripwire passes (Invariant 1).

**Tier budgets** (epoch caps unchanged): T1 = 5 epochs/15 min smoke, T2 = 30 epochs/120 min
standard, T3 = 50 epochs/240 min full. New experiment types (RL, fine-tuning stages) must
declare a tier when registered. **The minute caps are A40-calibrated** — on the local GTX
1650 they will kill runs prematurely, so always override with an explicit `--budget-minutes`
sized to measured throughput (§0.5), keeping any single run under ~8 h.

---

## 2. Current State (June 2026)

- **Champion: AR-041** (`ar-041_t2_curr_w5_action_attn_s2`) — 67.79% top-1, 93.42% top-3,
  move acc 71.86%, switch acc 60.51%. Config: 5L/256d/4H (~5.65M params), window=5,
  `split_head` + `action_self_attention` + `move_identity` + `shuffle_moves`, batch 1024,
  LR 4e-4, Elo-curriculum stage 2 (resume from stage 1).
- **Anchor (frozen reference):** P8-Lean 50K — 63.21% / 89.27%.
- **Established lessons (from 44 registered experiments):**
  - Window expansion requires capacity scaling (AR-004 failure).
  - Candidate/split policy heads were the single biggest architectural win (+3.4pp, +0.8pp).
  - Elo curriculum (stage 1 mid-Elo → stage 2 high-Elo) is worth multiple points.
  - Action self-attention adds relative reasoning (+1.3pp).
  - Batch >1024 has diminishing returns; value head at weight 0.1 *costs* ~0.8–1.6pp top-1
    (AR-036/037) — it must justify itself via win rate later, not top-1.
  - Switch prediction (60.5%) remains ~11pp behind move prediction — biggest offline lever.
- **Known defects:** aux speed/role/move-family heads receive zero gradient
  (`important_fixes/001`); checkpoints don't persist policy-head config flags
  (`important_fixes/002`) so `eval_harness.py` cannot reload AR-041; systematic
  overconfidence in the 0.4–0.8 confidence band.
- **Data:** 100K rated battle-perspectives processed locally (75,854 unique battles; every
  Metamon gen3ou battle ≥1500 Elo is already local). Remaining Metamon pool (~134K in
  1000–1500 bins) is of **unverified rating provenance** — many are likely unrated. See
  Invariant 3.
- **Infrastructure gaps:** no online evaluation of any kind (the Showdown stack —
  `showdown_client.py`, `battle_env.py`, `battle_evaluator.py`, `src/bots/` — exists only
  in the Pokemon-Battle-Model repo); no Gen 3 OU team files anywhere; no self-play
  recorder; no RL losses.
- **Research grounding (Metamon, arXiv:2504.04395 — the project our dataset comes from):**
  advantage-filtered BC beat pure BC; *diverse/unrealistic* self-play teams generalized
  better than realistic ones (their largest single jump: GXE 41–58% → 64–80%); two-hot
  value classification beat regression; long context was critical. Their 200M model reached
  top-10% of the ladder. Foul Play (search-based bot, ~1600–1800 Gen 3 OU) is the natural
  external baseline.

---

## 3. Phase A — Foundation Repairs & Rigor (BLOCKING — complete before new experiments)

| ID | Task | Details | Done when |
|---|---|---|---|
| A1 | Fix checkpoint head-flag persistence | `important_fixes/002`. Add `use_split_head`, `use_candidate_head`, `move_identity_candidates`, `policy_head_layers`, `action_self_attention`, `switch_weight`, `label_smoothing`, `max_seq_len` to `save_checkpoint` in `scripts/train_phase4.py` and to `load_checkpoint` in `Autoresearch/eval_harness.py`. Re-save champion weights with full config. | `eval_harness.py` loads and reproduces AR-041's registry metrics. |
| A2 | Fix aux-head missing targets | `important_fixes/001`. Wire `build_auxiliary_targets()` (`src/data/auxiliary_labels.py`) into the training data path; pass all four target tensors in `forward_step()`. | Validation logs show `aux_speed_accuracy > 0` and `aux_role_accuracy > 0`. |
| A3 | Establish the noise floor | Re-run the exact champion config with 3 seeds (T2). Record mean ± σ for top-1, switch acc, ECE in the registry notes. | Noise floor documented; future deltas interpreted against it. |
| A4 | Aux-weight ablation (post-fix) | With working aux heads: `aux_weight` ∈ {0, 0.1, 0.2, 0.4}. Hypothesis: real hidden-info gradient now helps the policy (it never could before A2). | Best aux weight identified; promote if > noise floor. |
| A5 | Calibration track | Label-smoothing sweep {0.0, 0.05, 0.1}; post-hoc temperature scaling on val; report ECE overall and by game phase (early/mid/late). | ECE reduced without top-1 regression; temperature constant stored with checkpoint. |

---

## 4. Phase B — Imitation-Learning Maximization

Goal: squeeze everything available out of supervised learning on rated human replays before
adding new learning signals. Run as independent single-variable experiments branching from
the current champion; stack winners.

### B-SW: Switch-prediction attack (biggest known gap)
| ID | Experiment | Config delta | Tier | Success criterion |
|---|---|---|---|---|
| B-SW-1 | Switch loss weight sweep | `switch_weight` ∈ {1.5, 2.0, 3.0} | T2 | Switch acc ↑ ≥2pp, top-1 not ↓ > noise |
| B-SW-2 | Switch-turn oversampling | Duplicate switch-decision examples in sampler (dataset-side alternative to B-SW-1) | T2 | Same as B-SW-1; compare vs B-SW-1 |
| B-SW-3 | "Will-switch" auxiliary head | New binary aux target: did the player switch this turn; small weight (0.05–0.1) | T2 | Switch acc ↑; aux head AUC > 0.75 |
| B-SW-4 | Explicit matchup features | Add type-effectiveness of opponent-active vs each own bench mon (computable from revealed info only — doctrine-safe) to per-pokemon continuous features | T2→T3 | Switch acc ↑ ≥2pp |
| B-SW-5 | Attention pooling for policy summary | Replace mean pooling with learned attention pooling before the policy head | T2 | Top-1 ↑ > noise |

### B-DATA: Data quality & scale (rated games only — Invariant 3)
| ID | Experiment | Config delta | Tier | Success criterion |
|---|---|---|---|---|
| B-DATA-1 | **Rating provenance audit** (prerequisite) | Inspect Metamon metadata for the 1000–1500 bins: distinguish genuinely-rated battles from unrated/default-Elo. Produce `data/rated_manifest.json`. | T1 (CPU) | Documented count of confirmed-rated battles available |
| B-DATA-2 | Rated-data expansion | Download + process confirmed-rated battles only (`download_replays_stratified.py`, `process_dataset.py`); retrain champion config | T3 | Top-1 ↑ > noise on the *fixed* current test split |
| B-DATA-3 | 3-stage Elo curriculum | Stage 1 (mid-Elo) → stage 2 (1300+) → stage 3 (1500+ only) via `resume_from` + `battle_manifest` | T3 | Beats 2-stage curriculum champion |
| B-DATA-4 | Elo-conditioning token | Add binned-Elo embedding to the context token; train on all rated data, condition on the top bin at inference | T2 | Top-1 ↑; high-Elo-conditioned eval beats unconditioned |
| B-DATA-5 | Winner-perspective weighting | Upweight (or restrict to) the winning player's actions; losers' mistakes are weaker supervision | T2 | Top-1 ↑ or gauntlet win rate ↑ (post-C) |
| B-DATA-6 | Elo-weighted loss | Per-example loss weight proportional to player rating | T2 | Compare vs B-DATA-3/4; keep best single mechanism |
| B-DATA-7 | Temporal split & holdout | Build a date-based test split; measure generalization across metagame drift | T1 (eval only) | Temporal-holdout metric added to eval harness |

### B-ARCH: Architecture
| ID | Experiment | Config delta | Tier | Success criterion |
|---|---|---|---|---|
| B-ARCH-1 | Window 7 with scaled capacity | `max_window=7`, 5–6L/288–320d (AR-004's lesson: window needs capacity) | T3 | Top-1 ↑ > noise |
| B-ARCH-2 | Window 10 (only if B-ARCH-1 wins) | `max_window=10`, scale model accordingly | T3 | Monotone improvement with window |
| B-ARCH-3 | 6L retry with depth-aware schedule | 6 layers + longer warmup (≥300), depth-scaled LR, pre-norm init check (AR-043 failed with stage-2 hyperparams) | T2 | Beats 5L at matched budget |
| B-ARCH-4 | Richer move-feature embeddings | Concatenate move metadata (base power, accuracy, type, category, priority/effect flags) onto the identity embedding — deepens Invariant 1 | T2 | Top-1 ↑; better off-meta generalization (post-C5) |
| B-ARCH-5 | Relative turn encoding | Replace sinusoidal turn encoding with ALiBi/rotary-style relative encoding | T2 | Top-1 ↑, esp. late-game accuracy |
| B-ARCH-6 | Memory tokens | A few learned tokens carried across the window to summarize pre-window history | T3 | Late-game accuracy ↑ |

### B-OPT: Optimization
| ID | Experiment | Config delta | Tier | Success criterion |
|---|---|---|---|---|
| B-OPT-1 | EMA of weights | Exponential moving average (decay 0.999) evaluated instead of raw weights | T2 | Top-1 ↑ ~0.3–0.5pp (typical) |
| B-OPT-2 | Stochastic weight averaging | SWA over final 25% of epochs | T2 | Compare vs B-OPT-1, keep one |
| B-OPT-3 | Cosine restarts / longer schedule | `cosine_epochs` sweep with warm restarts | T2 | Top-1 ↑ > noise |
| B-OPT-4 | Layer-wise LR decay | Lower LR for embeddings/early layers during curriculum stage 2 fine-tune | T2 | Stage-2 jump ↑ |

### B-ENS: Ensemble & distillation
| ID | Experiment | Config delta | Tier | Success criterion |
|---|---|---|---|---|
| B-ENS-1 | 3-seed logit ensemble | Average logits of 3 champion-config seeds (eval-only) | T1 (eval) | Ensemble top-1 ↑ ≥1pp over single |
| B-ENS-2 | Distill ensemble → single model | KL distillation from ensemble logits as auxiliary target | T3 | Single model recovers ≥50% of ensemble gain |

**Phase B exit gate:** offline metrics plateau (3 consecutive well-chosen experiments inside
noise floor) **or** Phase C is ready — whichever first. Do not chase decimals offline when
win-rate evaluation is available.

---

## 5. Phase C — Evaluator Buildout (gates everything downstream; start alongside late Phase B)

The single biggest process risk today is optimizing offline accuracy without knowing if the
model wins games. Build this before synthetic/RL work.

| ID | Task | Details |
|---|---|---|
| C1 | Port the online stack | Copy `src/environment/{showdown_client,battle_env,state,protocol,legality}.py`, `src/bots/`, `src/evaluation/` from Pokemon-Battle-Model (data layer is byte-identical; safe). Adapt `model_bot.py` to this repo's `BattleTransformer`: load with full head flags (post-A1) and add a **rolling window-5 observation buffer** (AR-041 was trained windowed; single-turn inference is a distribution shift). Validate live-obs fidelity vs replay obs — fix the hardcoded `forced_switch=False` and empty opponent base_stats/types in `_state_to_turn_obs`. |
| C2 | Local Showdown server | Run `setup_showdown.sh` (Node v22 present), smoke-test with the `run_phase1_exit_gate.py` pattern (`node pokemon-showdown start --no-security`), 100 RandomBot-vs-MaxDamageBot gen3ou games. |
| C3 | Gen 3 OU team pool | Author ~20 packed-format teams across archetypes: TSS, bulky offense, hyper offense, stall, rain (Kyogre-less ADV rain), Baton Pass. Include a **mirror-team mode** (both sides identical) to isolate decision quality from matchup luck. Store in `data/teams/gen3ou/`. |
| C4 | The Gauntlet | Fixed opponent suite: RandomBot, MaxDamageBot, HeuristicBot, frozen anchor checkpoint, frozen AR-019, frozen AR-041. ≥400 battles per opponent, alternating sides, shared team pool. Report win rate ± Wilson 95% CI. New `Autoresearch/run_online_eval.py` + a registry-patching utility so win rates are recorded in `experiment_registry.json`; `leaderboard.py` gains a gauntlet column. |
| C5 | Brutal extensions | Archetype-stratified win rates (no archetype below 40%); off-meta legal-set stress tests; temporal-holdout replay set (B-DATA-7); hidden-info calibration suite (score aux predictions vs eventual reveals); the shuffled-moveset tripwire as an automated check. |
| C6 | External baseline | Series vs Foul Play (open-source search bot). Losing is expected initially; track the trend per major version. |
| C7 | Ladder protocol | Readiness checklist: 100% legal-action rate over 1K local games, timeout/disconnect handling, choice-lock handling verified. Then: registered account, ≥400 ladder games per major version, GXE/Glicko recorded in the registry. **This is the project's final scoreboard.** |

**Phase C exit gate:** gauntlet runs end-to-end from a registry entry; AR-041 gauntlet
baseline recorded; ladder checklist items all green (ladder play itself can start at the
first major version after D or E).

---

## 6. Phase D — Synthetic Fine-Tuning (targeted repair, not a second pretraining)

Adapted from Pokemon-Battle-Model `docs/IMPLEMENTATION_PLAN.md` Phase 5. Exists only to
patch *identified* weaknesses.

- **D1 — Failure audit:** 500+ gauntlet/ladder losses categorized: mechanics errors, missed
  forced KOs, bad switch decisions, endgame conversion failures, hidden-info overcommit,
  rare-set collapse. Rank by frequency × severity.
- **D2 — Scenario factory** (`src/synthetic/`): perturb real replay anchor states (HP
  buckets, status toggles, hazard/weather toggles, usage-plausible bench swaps; never
  format-illegal states). Three label types only: hard labels (genuinely forced lines),
  acceptable-action sets (ranking loss / label smoothing across the set),
  opponent-distribution targets. A scenario enters training only if its label is stable
  across opponent-policy assumptions, seeds, and damage-roll perturbations.
- **D3 — Fine-tune:** existing `resume_from` + `battle_manifest` path; 70–80% real / 20–30%
  synthetic; 1–3 epochs; hard regression guard on real-replay validation (stop or reduce
  synthetic weight if it degrades).
- **D4 — Hard-example mining (cheap, do first):** upweight replay turns where the champion
  is confidently wrong vs the high-Elo player's action. No scenario factory needed.

**Exit gate:** targeted failure rates measurably drop with no general regression and no new
failure modes. If the gate fails, roll back — never force synthetic data harder.

---

## 7. Phase E — Offline RL (conditional)

**Promotion gate (ALL required):** BC+synthetic champion beats HeuristicBot ≥70% on the
gauntlet, archetype-robust (C5), aux heads usefully calibrated, no off-meta collapse.
If any gate fails, fix data/observation/eval instead — do not add RL.

The reward-adjacent signal already exists: `game_result` (win/loss) is stored per turn in
every `.npz` and plumbed through `WindowedTurnDataset` → `collate_windowed` →
`compute_total_loss`. All RL losses are new code in `forward_step` /
`compute_total_loss` plus new `run_experiment.py` config keys
(`ALLOWED_CONFIG_KEYS`, `build_train_command`, `validate_config`).

| ID | Experiment | Details | Success criterion |
|---|---|---|---|
| E1 | Two-hot value classification | Replace scalar value regression with two-hot classification over outcome bins (Metamon's finding); terminal win/loss target, optional tiny HP-differential shaping (off by default) | Value calibration (Brier) ↑ vs scalar head; needed by E2 and G2 |
| E2 | Advantage-filtered BC (binary) | Train critic, then weight BC loss by 𝟙[A(h,a) > 0] — Metamon's most effective simple variant | Gauntlet win rate ↑ vs BC+synthetic champion; offline metrics within guard |
| E3 | Exponential advantage weighting | AWR-style w = exp(β·A), β swept | Compare vs E2; keep one |
| E4 | Binary + MaxQ | Add λ·E_π[Q] term to E2's winner | Gauntlet ↑ further; watch calibration |
| E5 | Return-conditioned branch | Win-token conditioning (decision-transformer-lite): condition on "winner" at inference | Alternative if E2–E4 stall |

**Kill criteria** (from Model repo Phase 7): RL that improves in-distribution but worsens
robustness or calibration → roll back. RL gains smaller than remaining BC ablation gains →
deprioritize RL, reinvest in data/features.

---

## 8. Phase F — Self-Play & Population Training

- **F1 — Episode recorder (infrastructure):** accumulate per-turn `TurnObservation`s +
  chosen actions + outcome during `BattleEvaluator` games; `tensorize_battle` with frozen
  vocabs (`build_vocab=False`); `np.savez_compressed` in the existing key schema
  (`own_team, opponent_team, field, context, legal_mask, action, game_result, seq_len`).
  **Fidelity validation:** a model trained on *recorded* replays of human-replay games must
  match replay-trained metrics before any self-play data is trusted.
- **F2 — Diverse-team self-play:** generate 50–100K self-play battles using a deliberately
  **diverse team set** — Metamon found unrealistic-but-diverse teams generalize better to
  real opponents than realistic self-play. Self-play data is exempt from Invariant 3 (it
  has no Elo) but never replaces rated human data for the IL loss — it feeds RL/fine-tuning
  mixes only.
- **F3 — Mixed fine-tuning:** human + self-play mixture (start 80/20); cross-play
  evaluation vs a frozen checkpoint population to detect meta collapse (a new checkpoint
  must beat *old* checkpoints, not just its training partner).
- **F4 — League/PSRO-lite:** maintain 3–5 frozen opponents spanning styles; train vs the
  mixture; track exploitability via best-response probes.

---

## 9. Phase G — Inference-Time Improvements (cheap, orthogonal; any time after C)

| ID | Idea | Needs |
|---|---|---|
| G1 | Action-selection policy tuning: argmax vs low-temperature sampling, possibly per game phase | C4 |
| G2 | Top-k + value reranking: policy proposes top-3, calibrated value head scores 1-turn shallow rollouts, rerank | E1 (calibrated value) |
| G3 | Belief feedback: feed aux-head outputs (predicted items/speed/roles) back into the policy as features | A2, A4 |
| G4 | Inference ensemble of seeds (if ladder time controls allow) | B-ENS-1 |

---

## 10. Decision Gates Summary

| Transition | Gate |
|---|---|
| A → B | A1+A2 fixed and verified; noise floor recorded (A3) |
| B → C (parallel OK) | C work starts once B-SW and B-DATA-1 are underway; C must finish before D |
| C → D | Gauntlet operational; AR-champion baseline win rates recorded |
| D → E | Champion beats HeuristicBot ≥70%; archetype-robust; aux calibrated; no off-meta collapse |
| E → F | RL (or BC+synthetic, if RL killed) champion stable on gauntlet; episode recorder fidelity-validated |
| Any → Ladder | C7 checklist green; ladder runs at every major version thereafter |

## 11. Registry Conventions

- Experiment IDs continue sequentially (AR-045, AR-046, …) with descriptive slugs; tag the
  plan ID (e.g., `b-sw-1`) in the slug and the hypothesis field.
- Every entry: hypothesis, parent, single variable changed, tier, seeds, decision
  (KILL/RETRY/PROMOTE) with one-sentence justification.
- Win rates (post-C) are patched into the registry entry via the C4 utility; the
  leaderboard ranks by gauntlet score once available, offline top-1 before that.
- Multi-seed promotions: record mean ± σ, not best seed.

## 12. References

- Metamon: *Human-Level Competitive Pokémon via Scalable Offline RL with Transformers*
  (arXiv:2504.04395) — dataset source; offline RL & synthetic self-play findings.
- Foul Play (github.com/pmariglia/foul-play) — external search-bot baseline.
- Pokemon-Battle-Model `docs/IMPLEMENTATION_PLAN.md` Phases 5–8 — original designs for
  synthetic repair, brutal evaluation, conditional RL, and self-play, which Phases C–F
  here adapt.
