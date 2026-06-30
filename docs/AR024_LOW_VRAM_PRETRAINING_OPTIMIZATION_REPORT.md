# AR-024 low-VRAM pretraining optimization report

Date: 2026-06-30 (rev. 2 — deep code audit + compute-ceiling analysis)

## Executive summary

The goal: take AR-024-style champion-architecture pretraining on **100,000 battles** from
**~35 min/epoch to ~15 min/epoch** inside a **4 GB GTX 1650 (Turing, compute 7.5)** envelope
while keeping validation/test loss and top-1 within seed noise.

That is a **2.3× per-epoch speedup** (≈652 → ≈1,520 examples/s on ~1.37M train examples).
After auditing `scripts/train_phase4.py`, `src/models/battle_transformer.py`, and the
A40 reference reports, the honest conclusion is:

> **No single lever gets there. The 15-minute target is reachable only by stacking three
> independent classes of win — and the optimizer swap (Muon/AdEMAMix) is the *least*
> important of the three for the literal per-epoch metric.**

The three classes, in priority order:

1. **Runtime/IO hygiene (~1.2–1.4×, near-zero accuracy risk).** The training loop has
   real, fixable inefficiencies that are invisible on an A40 but bite hard on a 4 GB
   Turing card: **6–8 GPU→CPU synchronizations per micro-batch** from `.item()` calls in
   `compute_total_loss` and the metric accumulator, a **pure-Python padding collate**, a
   **per-example `clone()`** in the move-shuffle path, **plain ungrouped/un-fused AdamW**,
   and **whole-model `torch.compile` over variable-length batches** (which forces
   recompilation). Fixing these costs nothing scientifically.
2. **Static shapes + CUDA graphs (~1.2–1.5× on this model class).** The model is *tiny*
   (5.65M params) and the GPU is small, so **kernel-launch and Python overhead dominate**,
   not matrix FLOPs. Pad every window to a fixed length and the run becomes graph-capturable
   (`torch.compile(mode="reduce-overhead")` / CUDA graphs), which is the single largest
   runtime win available for a model this size.
3. **Per-example compute reduction (~1.4–1.8×, accuracy-gated).** The 1650 is genuinely
   **compute-bound** on the full champion arch (see §2). To move the per-epoch number you
   must reduce FLOPs/example: FFN ×3→×2, the action-self-attention ablation, and — the
   biggest single lever, but the most accuracy-sensitive — history window 5→4→3 (encoder
   self-attention is **quadratic in window**: 70²→42² tokens).

Muon and AdEMAMix change **time-to-target-loss** (fewer epochs/steps to the same val loss),
not per-epoch wall clock. They are the right bet for *total* research-loop throughput and
should be tested — but they do not, by themselves, move the user's stated per-epoch metric.
Frame and measure them accordingly.

**Recommended path to 15 min/epoch:** Stage 0 instrument → Stage 1 kill the syncs + fused
AdamW + larger micro-batch (lands ~25 min) → Stage 2 static shapes + CUDA graphs (lands
~19–21 min) → Stage 3 FFN ×2 and/or window 4 under accuracy guardrails (lands ~14–16 min).
Run Muon/AdEMAMix in parallel as an epochs-to-quality study, not as a per-epoch lever.

---

## 1. Current setup and AR-024 baseline

The repo is already ported for a 4 GB GTX 1650. The `gtx1650`/`local` profile
(`run_experiment.py`) and `--mode local` (`train_phase4.py`) encode: `batch_size=256`,
`grad_accum=4` (effective 1024), `amp=fp16` (Turing has 2× packed FP16 but **no bf16
hardware**), 5L/256d/window-5 champion arch, allocator `expandable_segments:True`,
`num_workers=6`, `prefetch_factor=4`, pinned memory, persistent workers, non-blocking H2D.

AR-024 as registered (`Autoresearch/notes/ar-024_…`, registry `status: halted`):

- Parent AR-020; 5.65M params; action self-attention + 2-layer split policy head; fp16.
- `batch_size=256 × grad_accum 4`; **25,000 battles** in the note (the 100K target in this
  report is the larger run the user wants to optimize); 10 epochs / `cosine_epochs=20`.

The script already has the low-VRAM primitives (allocator string set *before* `import torch`,
AMP context, GradScaler enabled only for fp16, DataLoader knobs, non-blocking transfer,
`--torch-compile` flag). The optimizations below are about *using them well* and removing
overhead that the A40 hid.

### A40 reference throughput (committed reports)

| Run | Arch / config | Batch/eff | Avg epoch | Avg ex/s | Test loss / top-1 |
|---|---|---:|---:|---:|---:|
| AR-020 S2 | 5L/256d, FFN×3, split head, action-attn, **window 5** | 1024/1024 | 242.8 s | 6,911 | **0.8455 / 67.79%** |
| AR-023 S1 | AR-020, batch 4096, LR 8e-4 | 4096/4096 | 127.2 s | 7,230 | 1.0292 / 60.11% |
| AR-002 | window **2**, 2.33M params | 1024/1024 | 91.4 s | 15,150 | 1.0741 / 58.33% |
| AR-005 | 5L/256d, window 2, 4.33M params | 1024/1024 | 97.3 s | 14,237 | 1.8053 / 27.0% |

Two things to read from this table:
- **Window dominates throughput.** Window-2 runs hit ~15K ex/s; the window-5 champion
  drops to ~6.9K ex/s on the *same* A40 — a ~2.2× slowdown purely from the longer token
  sequence (attention is quadratic; see §2/§5). This is the lever with the most headroom
  and the most accuracy risk.
- **AR-023 (batch 4096) is a cautionary tale:** bigger batch sped the A40 up but *lost
  6.7 pts of top-1*. Effective batch is a science variable here, not a free speed knob —
  keep effective batch ≈1024 unless a paired run proves loss is preserved.

---

## 2. Hardware reality check: the 1650 is compute-bound on this arch

The previous revision framed the target as "a utilization problem." The deeper truth is
that it is **mostly a compute-ceiling problem**, and naming that changes the plan.

Rough FP16 throughput: A40 runs bf16/FP16 on tensor cores at ~150–300 TFLOPS effective;
the GTX 1650 (TU117) has **little/no usable tensor-core path for this workload** and runs
packed-FP16 on CUDA cores at ~10–12 TFLOPS. For a 5.65M-param model the A40 is *not* tensor
saturated (it's partly launch/memory bound at this size), so the realized ratio is smaller
than peak — empirically ~8–11×.

- A40 champion arch: ~6,900 ex/s.
- Implied 1650 compute ceiling: ~6,900 / (8–11) ≈ **630–860 ex/s** for the *unchanged* arch.
- User's observed local rate: 35 min on ~1.37M ex ≈ **652 ex/s** — i.e., the current run is
  **already near the 1650's compute ceiling for the full champion arch.**

**Implication:** runtime/IO hygiene (§3) recovers the gap between "near ceiling" and "at
ceiling" — maybe pushing 652 → ~800–860 ex/s (≈27 → ~26 → ~25 min). That is real but it is
**not** 2.3×. The remaining factor of ~1.8× must come from **doing less work per example**
(§5) and/or **removing launch overhead via CUDA graphs** (§4). Convergence optimizers (§6)
do not change ex/s at all.

This is the single most important correction to the plan: **budget the speedup as a product
of (IO hygiene) × (graph/launch reduction) × (FLOP reduction), and accept that hitting 15
minutes requires touching the architecture's per-example compute, under accuracy guardrails.**

---

## 3. Code-level throughput findings (audit of `train_phase4.py`)

These are concrete, low/zero-risk defects found by reading the training loop. Each is a
near-free win on the 1650.

### 3.1 6–8 GPU→CPU syncs per micro-batch (highest-value runtime fix)

`compute_total_loss` (`battle_transformer.py:1357–1400`) calls `.item()` on **policy, each
aux head, auxiliary total, value, and total** — every call is a forced
`cudaStreamSynchronize`. `train_epoch` (`train_phase4.py:497–500`) adds **three more**
(`total_correct`, `total_top3`, `total_examples`) inside the loop. At `grad_accum=4` that is
~28–32 device syncs per optimizer step, each stalling the pipeline and preventing
backward/H2D overlap. On an A40 these hide under compute; on a 1650 they are a measurable
slice of the 35 minutes.

**Fix:** return loss components as **tensors**, accumulate them on-GPU, and `.item()` **once
per epoch**. For accuracy counters, keep running GPU tensors (`total_correct += (...).sum()`
without `.item()`) and sync once at epoch end. Expected: 5–15% on the 1650; it is the first
thing to do after instrumentation.

### 3.2 Pure-Python padding collate (`collate_windowed`, lines 297–327)

The collate does a Python `max()` over the batch, then a per-item Python loop building
`torch.zeros` pads and `torch.cat`s, then `torch.stack`. This is host-bound work on the
DataLoader workers and produces **variable `max_len` per batch** (the root cause of the
`torch.compile` problem in §3.5). With 6 workers on a mid-range CPU it can starve the GPU.

**Fix options (cheap → involved):**
- **Pad to a fixed `max_window` (=5) always** and `drop_last=True`. Removes the Python
  `max()`/branching, makes every batch the same shape (enables §4), and the wasted compute
  on early-turn windows is small (window ≤5).
- Pre-allocate the output tensor once and index-copy instead of `cat` per item.
- For the selected 100K run, optionally **precompute windowed examples into a contiguous
  (memmap) tensor** so `__getitem__` is a pure slice — trades RAM (watch the 15 GB cap) for
  eliminating per-item host work. Gate behind a flag; report peak RAM.

### 3.3 Per-example `clone()` in move-shuffle (`_apply_move_shuffle`, lines 223–261)

For every `__getitem__`, when `shuffle_moves=True` the code clones `own_team`, `legal_mask`,
and `action`, then does `.item()`/`.nonzero()` to remap the label. Since the **permutation
is fixed per battle**, this work is redundant per turn.

**Fix:** precompute, once per battle at dataset build, the shuffled `own_team`/`legal_mask`
views and an `action`-remap lookup table; `__getitem__` then slices already-shuffled tensors
with no clone and no Python scalar ops. Keeps the invariant (one perm/battle, applied to all
turns) bit-for-bit. This offloads work from the hot path into one-time setup.

### 3.4 Plain ungrouped, un-fused AdamW (lines 1272–1273)

`torch.optim.AdamW(model.parameters(), …)` uses the **single-tensor** path and **one param
group**. On Turing/CUDA, `fused=True` runs the entire optimizer step in one kernel
(vs. hundreds of small launches across 5.65M params spread over many tensors); `foreach=True`
is the fallback. The optimizer step runs every `grad_accum` steps, so this is a steady win.

**Fix:** build AdamW with `fused=True`, catch `TypeError`/`RuntimeError` → `foreach=True` →
plain. Also split into **decay / no-decay param groups** (no weight decay on norms/biases/
embeddings) — standard, slightly better loss, and a prerequisite for the Muon hybrid (§6).

### 3.5 `torch.compile` applied to the whole model over variable shapes (lines 1233–1234)

`torch.compile(model)` with the current collate sees a **different `max_len` (1–5) almost
every batch**, so it either recompiles repeatedly or falls back with guards/graph breaks —
often *slower* than eager and memory-hungry on 4 GB. This is why the existing report (rev. 1)
hedged on compile.

**Fix:** combine with §3.2 (fixed window) so shapes are static, then compile with
`dynamic=False` and, once stable, `mode="reduce-overhead"` (CUDA graphs). See §4.

### 3.6 Minor

- `optimizer.zero_grad()` (line 484) → `zero_grad(set_to_none=True)` (explicit; default is
  already `True` in recent PyTorch, but be explicit and it composes better with fused).
- `total_steps = cosine_epochs * len(train_loader) // grad_accum` (line 1275): with
  `drop_last=False` and accumulation, the last partial accumulation window per epoch is
  dropped from the count — fine, but if you switch to `drop_last=True` re-derive this so the
  cosine schedule lands where intended.

---

## 4. Static shapes + CUDA graphs: the biggest runtime lever for a tiny model

Because the model is small, **per-kernel launch latency and Python dispatch dominate**
realized time on the 1650, not the GEMMs. Two compounding wins once shapes are static
(§3.2):

1. **`torch.compile(model, dynamic=False)`** — fuses pointwise ops, cuts dispatch overhead.
2. **`mode="reduce-overhead"` (CUDA graphs)** — captures the whole step as one graph,
   removing per-op launch latency. For 1–10M-param transformers this is commonly 1.3–2×.

Requirements/caveats: static batch and sequence shapes (`drop_last=True`, fixed window),
stable control flow, and a few warmup steps before timing. Validate peak VRAM still fits
4 GB (graph capture pins some memory). Accept only if measured steady-state epoch time
improves ≥10% with no OOM. **Note:** TF32 is an Ampere+ feature and is *not* available on
Turing — do not add `allow_tf32` expecting a speedup here.

---

## 5. Where the FLOPs go, and the per-example compute levers

The encoder runs self-attention over `TOKENS_PER_STEP = 14` tokens/step
(`battle_transformer.py:241`: 6 own + 6 opp + field + context), across the whole window:

- window 5 → **70 tokens**, attention ∝ 70² = 4,900
- window 4 → 56 tokens → 3,136 (0.64×)
- window 3 → 42 tokens → 1,764 (0.36×)
- window 2 → 28 tokens → 784 (0.16×)

This is exactly why the A40 dropped from ~15K ex/s (window 2) to ~6.9K (window 5). On the
1650, **window is the highest-leverage compute knob — and the highest accuracy risk** (the
champion is window-5 for a reason; window-2 runs collapsed to 27–58% top-1). Treat any
window reduction as a *science* experiment with paired val/test loss, not a free speedup.

Compute levers, ranked by (FLOP saved ÷ accuracy risk):

| Lever | FLOP effect | Accuracy risk | Notes |
|---|---|---|---|
| FFN ×3 → ×2 | cuts ~⅓ of FFN GEMMs (a large share of encoder compute) | **Low–moderate** | Best first compute cut; AR-008 tested FFN×4, never a clean ×2 ablation vs champion |
| action-self-attention off | removes a 9-candidate MHA in the head | Moderate | AR-019/020 suggest it *helps* top-1 — ablate carefully |
| window 5 → 4 | ~0.64× encoder attention | High | Paired loss check mandatory |
| window 5 → 3 | ~0.36× encoder attention | **Very high** | Only if FFN×2 + graphs miss target |
| layers 5 → 4 | ~0.8× encoder | High | Linear cut; champion depth is load-bearing |

Do **not** combine multiple compute cuts in one run — one variable per experiment, paired
against the AR-024 AdamW control, judged on test policy loss within +0.01–0.02 and top-1
within ~0.5 pt (see §8).

---

## 6. Optimizer research: Muon, AdEMAMix, Sophia

**Reframe up front:** these change *epochs/steps to a target loss*, not *seconds per epoch*.
Their payoff is **total time-to-quality** and possibly **fewer epochs at equal val loss** —
valuable for the research loop, but they do not by themselves hit the user's per-epoch
number. Measure them on "val loss at equal wall-clock" and "epochs to reach control's best
val loss," not on ex/s.

### Muon (Momentum Orthogonalized by Newton-Schulz)

Muon orthogonalizes matrix-shaped updates via a few Newton-Schulz iterations and has driven
NanoGPT/CIFAR speedrun records (<https://kellerjordan.github.io/posts/muon/>). It targets
exactly the dense 2D hidden weights that dominate AR-024 (encoder Q/K/V/O, FFN, policy/action
projections). On a 5.65M model the NS iterations are cheap (per-step overhead negligible),
and Muon carries **less optimizer state than Adam** for the matrices it manages — a minor
VRAM bonus (though optimizer state is already small here; activations are the VRAM ceiling).

Rules for this repo:
- **Muon only on 2D hidden matrices.** Keep **AdamW** for all embeddings (species/move/item/
  ability/type), LayerNorm/biases (1D), the final logit/scoring layers, and aux heads.
  Requires the param-group split from §3.4.
- Muon tolerates **larger effective batch** without much LR retuning — useful, since bigger
  batch = fewer steps. But effective batch is a *science* variable here (AR-023 lost 6.7 pt
  at batch 4096) — sweep batch *separately* from the optimizer.
- Tune Muon LR (~0.02 start) and keep the AdamW group at 4e-4; identical warmup-step count.

| ID | Muon group | AdamW group | LR plan | Accept |
|---|---|---|---|---|
| M0 | none | all | existing | control |
| M1 | encoder 2D matrices | embeds, norms, biases, all heads | Muon 0.02 / AdamW 4e-4 | ≤ control val loss by epoch 6–8 |
| M2 | encoder + policy 2D matrices | embeds, norms, biases, final logits | Muon 0.02–0.04 / 4e-4 | top-1 drop ≤0.5 pt |
| M3 | M2 @ eff-batch 2048 | same | √-scale AdamW to ~5.6e-4 | fewer epochs to control's best val loss |

### AdEMAMix

Adds a second, slow gradient EMA to Adam, reporting comparable quality with substantially
fewer tokens (<https://arxiv.org/abs/2409.03137>). Lowest-integration-risk Adam-family bet.
Cost: a **third momentum buffer** (+~22 MB optimizer state at 5.65M params) — trivial vs the
activation budget on 4 GB. Run as an ablation: start at AdamW LR, also try 0.75×/1.25×, keep
weight decay/warmup fixed, judge on **epochs-to-control-val-loss** and val loss at equal
wall-clock.

### Sophia (and other second-order/clipped methods)

Diagonal-curvature + clipped updates, ~2× step/compute/wall-clock for GPT pretraining
(<https://arxiv.org/abs/2305.14342>). **Defer.** For a 5.65M model whose batch is VRAM-capped
and whose bottleneck is launch/compute (not optimization convergence), the Hessian-estimator
overhead and tuning risk are poor trade-offs. Revisit only if §3–§5 are exhausted and
instrumentation shows the limiter is *convergence*, not throughput.

---

## 7. Recommended commands and engineering work order

### 7.1 AdamW control (run this first, unchanged science)

```bash
python scripts/train_phase4.py --mode local \
  --num-battles 100000 \
  --epochs 10 --cosine-epochs 20 \
  --split-head --move-identity --shuffle-moves --no-value-head \
  --ffn-multiplier 3 \
  --batch-size 256 --grad-accum 4 \
  --amp fp16 \
  --num-workers 6 --prefetch-factor 4 --persistent-workers --pin-memory \
  --report-path checkpoints/ar024_100k_adamw_control/training_report.json \
  --checkpoint-dir checkpoints/ar024_100k_adamw_control
```

### 7.2 Throughput candidate (after §3 fixes land)

Fewer optimizer steps + static shapes; fall back to `256×4` on OOM.

```bash
python scripts/train_phase4.py --mode local \
  --num-battles 100000 \
  --epochs 10 --cosine-epochs 20 \
  --split-head --move-identity --shuffle-moves --no-value-head \
  --ffn-multiplier 3 \
  --batch-size 384 --grad-accum 3 \
  --amp fp16 \
  --num-workers 6 --prefetch-factor 4 --persistent-workers --pin-memory \
  --torch-compile \
  --report-path checkpoints/ar024_100k_fast/training_report.json \
  --checkpoint-dir checkpoints/ar024_100k_fast
```

(Effective batch stays 1152 ≈ 1024-class; `512×2` only if profiling shows VRAM headroom —
on 4 GB at window 5 it is borderline.)

### Engineering work order

1. **Instrument first.** Add a `--benchmark N` mode that times, over 200–500 batches and
   reports to JSON: host collate, H2D copy, forward, backward, grad-clip, optimizer step,
   scheduler, validation, checkpoint, and **peak VRAM**. Nothing below is evaluated without
   this baseline — it tells you whether you are host-bound, transfer-bound, or compute-bound.
2. **Defer the `.item()` syncs (§3.1).** Tensors out of `compute_total_loss`; GPU-side metric
   accumulation; one sync/epoch.
3. **Fused/foreach AdamW + decay param groups (§3.4)** with safe fallback; log group sizes.
4. **`zero_grad(set_to_none=True)` (§3.6).**
5. **Fixed-window collate + `drop_last=True` (§3.2)**; re-derive `total_steps`.
6. **Precompute move-shuffle per battle (§3.3).**
7. **`torch.compile(dynamic=False)` → `mode="reduce-overhead"` (§4)**; accept on ≥10%
   measured steady-state gain, VRAM still <4 GB.
8. **DataLoader sweep** `num_workers∈{4,6,8}`, `prefetch∈{2,4}` for 500 batches; pick the
   fastest *stable* setting, not the most workers.
9. **Optimizer factory** `--optimizer {adamw,adamw_fused,ademamix,muon}` for §6 studies.
10. **Compute-reduction experiments (§5)**, one variable each, accuracy-gated.

---

## 8. Accuracy guardrails (unchanged — promotion gates)

A speed win is rejected unless, on the **same split/seed/`shuffle_moves=True`/val+test sets**:

- Test policy loss ≤ +0.01 (promotion) / +0.02 (trial) vs seed-matched AdamW control.
- Test top-1 within 0.5 pt of control.
- Switch-action accuracy not down >1.0 pt aggregate.
- ECE not materially worse; if confidence rises without accuracy, require temperature
  calibration before promotion (infra in `Autoresearch/calibration.py`).
- Shuffled-moveset tripwire passes (collapse ⇒ KILL — non-negotiable invariant).

---

## 9. Experiment ladder with throughput budget

Estimates assume the ~652 ex/s (≈35 min) starting point and compound multiplicatively;
treat them as hypotheses to confirm with the Stage-0 benchmark.

| Stage | Change | Class | Est. ex/s | Est. min/epoch | Accuracy risk | Stop/go |
|---|---|---|---:|---:|---|---|
| 0 | Instrument 200–500 batches | measure | 652 | 35 | none | have timing + peak VRAM |
| 1 | Defer `.item()` syncs + fused AdamW + set_to_none | IO/runtime | ~720–780 | ~30–32 | none | ≥5% faster batches |
| 2 | Fixed-window collate + precomputed shuffle + DataLoader sweep | IO/runtime | ~800–880 | ~26–28 | none | GPU util up, no RAM blowup |
| 3 | `batch 384×ga3` (eff ~1152) | runtime | ~860–950 | ~24–26 | low (eff-batch≈1024) | fits <3.7 GB, val loss unchanged |
| 4 | `torch.compile` + CUDA graphs (static shapes) | launch | ~1,050–1,300 | ~18–21 | none–low | ≥10% steady gain, no OOM |
| 5 | **FFN ×3→×2** | compute | ~1,250–1,500 | ~15–18 | **low–moderate** | test loss ≤ +0.02 |
| 6 | window 5→4 *(only if still short)* | compute | ~1,500–1,800 | ~13–15 | **high** | paired loss within gates |
| — | Muon / AdEMAMix (parallel track) | convergence | n/a (ex/s) | fewer epochs | research | ≤ control val loss in fewer epochs |

Reading: **Stages 1–4 (zero/low accuracy risk) plausibly reach ~18–21 min.** Hitting the
**15-min** target requires Stage 5 (FFN×2) and possibly Stage 6 (window 4) — i.e., a
deliberate, guarded per-example compute cut. That is the central trade the user is implicitly
asking for, and it should be made explicitly, with paired loss evidence, not silently.

---

## 10. Bottom line

The 35→15 min/epoch target is **achievable but not free**, and the previous framing
("utilization problem, Muon is the interesting bet") was half-right. The corrected picture:

- The 1650 is **already near its compute ceiling** on the unchanged champion arch, so
  runtime/IO hygiene alone (~1.3×) lands ~26–28 min, not 15.
- The **largest pure-runtime win is static shapes + CUDA graphs** (~1.3–1.5×) because this
  tiny model is launch-bound — but it requires the fixed-window collate fix first.
- Reaching 15 min **requires a per-example FLOP cut** (FFN×2, then window) under strict
  accuracy guardrails; this is a science decision, made one variable at a time.
- **Muon/AdEMAMix are a separate, worthwhile track** that improves *time-to-quality*
  (fewer epochs), not seconds-per-epoch — test them, but don't credit them against the
  per-epoch metric.

Most of the machinery (allocator tuning, AMP, GradScaler, DataLoader knobs, `--torch-compile`,
profile presets) already exists; the work is removing the hidden syncs, making shapes static,
adding a fused/grouped optimizer factory, and running the compute-reduction ladder with the
existing accuracy gates and shuffle tripwire.
