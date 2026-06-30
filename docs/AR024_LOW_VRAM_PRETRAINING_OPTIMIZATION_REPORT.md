# AR-024 low-VRAM pretraining optimization report

Date: 2026-06-30

## Executive summary

The target is aggressive but feasible if we treat it as a throughput engineering target rather than a pure optimizer swap: reduce AR-024-style pretraining on 100,000 battles from about 35 minutes per epoch to 15 minutes per epoch while staying inside a 4 GB GPU envelope and preserving validation/test loss. The current local recipe already ports the A40 champion constraints to GTX 1650-class hardware: `batch_size=256`, `grad_accum=4`, `fp16`, 25,000 battles, history window 5, split action head, move identity, shuffled moves, and no value head. AR-024 is explicitly a 5.65M-parameter champion-architecture noise-floor run derived from AR-020.

The key observation from prior runs is that AR-020/AR-023 on A40 achieve about 7,000-7,500 examples/s for the champion action-attention architecture, while the requested 100,000-battle/15-minute target implies roughly 6,100 examples/s if the 100K corpus scales from the AR-020 data density. Therefore, the 15-minute target is not algorithmically impossible; it is a local-hardware utilization problem. On a GTX 1650, the fastest route is to avoid increasing model size, keep the window at 5, reduce optimizer-update overhead, compile only if it survives warmup, and attack DataLoader/collate overhead.

Recommended order:

1. **Benchmark before changing science.** Add a 200-500 batch microbenchmark that separates host collation, H2D copy, forward/backward, optimizer step, validation, and checkpoint time. Do not evaluate optimizer changes without this baseline.
2. **Adopt a two-lane optimizer plan.** Keep AdamW as the control. Test Muon only on 2D hidden weights with AdamW for embeddings, biases, norms, and heads; test AdEMAMix as the lowest-risk AdamW-family alternative. Treat Sophia as a later experiment because Hessian-estimator overhead and tuning risk are high for this small model.
3. **Maximize effective throughput under 4 GB.** Prefer the largest stable micro-batch that fits (`512 x grad_accum 2` if headless; otherwise `256 x 4`), use fused/foreach AdamW where available, pinned memory, persistent workers, and non-blocking transfers.
4. **Reduce per-example compute only where accuracy evidence is strong.** Keep AR-024 architecture for the first speed pass. If still below target, evaluate `ffn_multiplier=2`, action-self-attention off/on ablation, and activation checkpointing only when batch size must be increased.
5. **Measure success against validation/test loss, not training loss.** A candidate is acceptable only if test policy loss remains within about +0.01-0.02 of AR-020/AR-024-class controls and top-1 does not degrade by more than seed noise.

## Current setup and AR-024 baseline

The repository is already tuned for a 4 GB GTX 1650 target. The README states the target hardware, notes that the A40 champion used `batch_size=1024` and `bf16`, and says those settings do not directly transfer to local hardware. The local port replaces that with micro-batch `256 x grad_accum 4`, `fp16`, allocator tuning, and smaller data scale.

AR-024 is defined as a local champion-architecture noise-floor experiment:

- Parent: AR-020.
- Architecture: 5.65M parameters, action attention, 2-layer policy head.
- Precision: `fp16`.
- Batch plan: `batch_size=256`, `grad_accum=4`, effective batch 1024.
- Data: 25,000 battles.
- Schedule: 10 epochs, `cosine_epochs=20`.
- DataLoader: `prefetch_factor=2`.

The training script supports the important low-VRAM primitives already: the CUDA allocator is set to expandable segments before importing PyTorch, AMP can be `fp16`/`bf16`/auto, the local mode sets 5L/256d/window-5 with `batch_size=256` and `grad_accum=4`, and DataLoader knobs include workers, pinned memory, persistent workers, prefetch factor, and non-blocking GPU transfer.

## Prior-run evidence from the repository

A40 reference results from the committed reports:

| Run | Architecture/config | Train examples | Batch/effective batch | Avg epoch time | Avg examples/s | Test loss/top-1 |
|---|---:|---:|---:|---:|---:|---:|
| AR-020 S1 v2 | 5L/256d, FFN x3, split head, move identity, action attention | 919,901 | 1024/1024 | 130.8 s | 7,035.6 | 1.0219 / 60.53% |
| AR-023 S1 | AR-020 with batch 4096 and LR 8e-4 | 919,901 | 4096/4096 | 122.7 s | 7,497.3 | 1.0292 / 60.11% |
| AR-021 S1 | 3-layer policy variant | 919,901 | 1024/1024 | 139.9 s | 6,577.5 | 1.0374 / 60.07% |

Interpreting the 100,000-battle target:

- AR-020 has 67,181 battles and 919,901 training examples, or about 13.69 train examples per battle.
- A 100,000-battle run would therefore be about 1.37M training examples, depending on split and filtering.
- 15 minutes/epoch requires about 1,520 examples/s if exactly 1.37M examples are used in 900 seconds. If the user-observed 35 minutes is the measured local epoch time, current local throughput is about 650 examples/s. The target is therefore about a 2.3x local speedup.
- A40 evidence shows the model/science can run much faster than the target; local hardware must be kept saturated and should avoid unnecessary optimizer-step and host-side overhead.

## Research scan: optimizer and speed techniques

### Muon optimizer

Muon (MomentUm Orthogonalized by Newton-Schulz) applies momentum and then orthogonalizes matrix-shaped updates with Newton-Schulz iterations. Keller Jordan's Muon writeup describes it as an optimizer for hidden layers and notes use in training-speed records for NanoGPT and CIFAR-10 speedrunning: <https://kellerjordan.github.io/posts/muon/>. The practical pattern in small transformer training is not to replace every parameter group. Use Muon for matrix weights in hidden layers, and keep AdamW for embeddings, biases, LayerNorm/RMSNorm parameters, output heads, and any 1D tensors.

Why it is promising here:

- AR-024 is dominated by dense linear projections in the transformer blocks and action/policy head, exactly the layer class Muon targets.
- Muon stores less Adam-style second-moment state for Muon-managed matrices, which can help 4 GB memory pressure.
- Faster convergence may allow fewer epochs at similar validation loss, which is more valuable than a small per-step speedup.

Risks:

- Muon is less established than AdamW and needs LR/momentum tuning.
- Newton-Schulz iterations add per-step work; on a small GTX 1650, wall-clock speed may not improve unless reduced epochs or larger batch compensate.
- Embeddings and heads should stay on AdamW to avoid destabilizing sparse/categorical parameters.

Recommended AR-024 Muon grid:

| ID | Muon parameter set | AdamW parameter set | LR plan | Acceptance criterion |
|---|---|---|---|---|
| M0 | none | all params | existing AdamW | control |
| M1 | encoder hidden 2D matrices only | embeddings, norms, biases, policy/action heads | Muon LR 0.02, AdamW LR 4e-4 | same/lower val loss by epoch 6-8 |
| M2 | encoder + policy hidden 2D matrices | embeddings, norms, biases, final logits | Muon LR 0.02-0.04, AdamW LR 4e-4 | no top-1 drop >0.5 pp |
| M3 | M2 with effective batch 2048 | same | sqrt-scale AdamW LR to 5.6e-4 | improves epoch time and val loss slope |

### AdEMAMix

AdEMAMix adds a second, slower EMA of gradients to Adam-like training. The paper reports faster convergence and, in one LLM example, comparable performance with substantially fewer tokens than AdamW: <https://arxiv.org/abs/2409.03137>. It is a good candidate because it is close to AdamW operationally and should be simpler to integrate than Hessian-based methods.

Recommended use:

- Run as an optimizer-ablation, not a default.
- Start from AdamW LR, then try 0.75x and 1.25x LR.
- Keep weight decay and warmup identical.
- Judge by validation loss at equal wall clock and equal example count.

### Sophia / second-order clipped optimizers

Sophia estimates diagonal curvature periodically and clips updates. The ICLR paper reports roughly 2x speed-up in steps/compute/wall-clock for GPT pretraining: <https://arxiv.org/abs/2305.14342>. For AR-024 it is lower priority than Muon/AdEMAMix because the model is small, batches are constrained by VRAM, and Hessian-estimator overhead can eat the win. It is worth testing only after instrumentation shows the local bottleneck is optimization convergence rather than input pipeline or GPU occupancy.

### PyTorch runtime techniques

PyTorch's performance tuning guide recommends practical levers such as asynchronous/pinned input transfer, avoiding unnecessary synchronization, tuning DataLoader workers, setting gradients to `None`, mixed precision, channels/memory layout where relevant, and compilation/fusion where appropriate: <https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>. For this repo, the highest-value items are:

- Use `optimizer.zero_grad(set_to_none=True)` rather than the current default zeroing.
- Try `torch.optim.AdamW(..., fused=True)` on CUDA builds that support it; fallback to `foreach=True`; otherwise standard AdamW.
- Keep `pin_memory=True`, `persistent_workers=True`, and non-blocking transfer.
- Autotune `num_workers` and `prefetch_factor`; the current local AR-024 note uses `prefetch_factor=2`, while A40 champion reports used 4.
- Test `torch.compile` only after a short warmup; compile can help steady state but can be too expensive or memory-hungry on 4 GB cards.

## Recommended 4 GB AR-024 training recipe

### Baseline control command

Use this as the reproducible control for every experiment:

```bash
python scripts/train_phase4.py --mode local \
  --num-battles 100000 \
  --epochs 10 --cosine-epochs 20 \
  --split-head --move-identity --shuffle-moves --no-value-head \
  --ffn-multiplier 3 \
  --batch-size 256 --grad-accum 4 \
  --amp fp16 \
  --num-workers 4 --prefetch-factor 4 --persistent-workers --pin-memory \
  --report-path checkpoints/ar024_100k_adamw_control/training_report.json \
  --checkpoint-dir checkpoints/ar024_100k_adamw_control
```

### Throughput-first command candidate

If VRAM is headless and stable, reduce optimizer steps per epoch with a larger micro-batch:

```bash
python scripts/train_phase4.py --mode local \
  --num-battles 100000 \
  --epochs 10 --cosine-epochs 20 \
  --split-head --move-identity --shuffle-moves --no-value-head \
  --ffn-multiplier 3 \
  --batch-size 512 --grad-accum 2 \
  --amp fp16 \
  --num-workers 4 --prefetch-factor 4 --persistent-workers --pin-memory \
  --report-path checkpoints/ar024_100k_bs512_ga2/training_report.json \
  --checkpoint-dir checkpoints/ar024_100k_bs512_ga2
```

Expected effect: fewer accumulation loops, fewer optimizer/scheduler steps, better GPU occupancy. If this OOMs, return to `256 x 4`; if it fits but lowers validation quality, keep effective batch 1024 and do not increase further.

### Experimental Muon candidate

Implement parameter grouping before running this. Suggested default:

- Muon on hidden 2D matrices in encoder blocks and non-final policy MLP/attention projections.
- AdamW on embeddings, biases, LayerNorm, final scoring heads, and auxiliary heads.
- Keep effective batch 1024 for the first Muon run.
- Warm up for the same number of optimizer steps as AdamW.

Expected effect: possibly same validation loss in fewer epochs. Do not assume per-epoch wall clock improves; the win may be fewer required epochs.

## Concrete engineering changes to prioritize

1. **Add a benchmark mode/report section.** Record per-epoch and per-N-batch timings for `collate`, H2D transfer, forward, backward, clip, optimizer step, scheduler, validation, checkpoint save, and GPU peak memory. This is required to know whether the 35-minute epoch is CPU-bound, transfer-bound, or compute-bound.
2. **Optimizer factory.** Add `--optimizer {adamw,adamw_fused,ademamix,muon}` with safe fallback. For Muon, add explicit parameter grouping and log counts by group.
3. **Use zero-grad set-to-none.** Replace `optimizer.zero_grad()` with `optimizer.zero_grad(set_to_none=True)` in the training loop.
4. **Fused AdamW fallback.** Try `fused=True` on CUDA, catch `TypeError` at construction, then fallback to `foreach=True`, then standard AdamW. This is a low-risk speed improvement.
5. **DataLoader sweep.** Run `num_workers={2,4,6,8}` and `prefetch_factor={2,4}` for 500 batches. Pick the best stable setting, not the most workers.
6. **Compiled model trial.** Run 1 warmup epoch and 2 measured epochs with `--torch-compile`. Accept only if measured epoch time improves by at least 10% and peak VRAM still fits.
7. **Collate optimization.** If benchmark shows host-bound behavior, precompute padded window tensors or cache per-example windows for the selected 100K run. This trades RAM for speed; on a 15 GB RAM target, keep cache optional and report peak RAM.
8. **Validation cadence.** For speed studies, validate every epoch for science runs, but for hyperparameter sweeps validate every 2 epochs plus final. Validation does not affect training epoch time but affects wall-clock experiment throughput.
9. **Checkpoint cadence.** Keep best checkpoint, but save rotating epoch checkpoints less frequently during speed sweeps. Checkpoint I/O can distort local wall-clock on slow disks.
10. **Architecture fallback.** If throughput remains below target, run `ffn_multiplier=2` and compare to AR-020/AR-024 validation loss. FFN x3 is a likely compute hotspot; reducing it is less disruptive than shrinking history window or disabling move identity.

## Accuracy guardrails

Do not accept a speed win unless it passes these gates:

- Test policy loss no worse than +0.01 for a promotion candidate or +0.02 for a trial candidate.
- Test top-1 no worse than 0.5 percentage points below seed-matched AdamW control.
- Switch-action accuracy not degraded by more than 1.0 percentage point on aggregate.
- Calibration/ECE not materially worse; if confidence increases while accuracy does not, require temperature calibration before promotion.
- Same data split, same seed, same `shuffle_moves=True` training invariant, and same validation/test sets.

## Experiment ladder to hit 15 minutes/epoch

| Stage | Change | Why | Stop/go criterion |
|---|---|---|---|
| 0 | Instrument 500 batches | Locate bottleneck | Have timing and peak VRAM breakdown |
| 1 | `zero_grad(set_to_none=True)` + fused/foreach AdamW | Low-risk runtime speed | >=5% faster measured batches |
| 2 | DataLoader sweep | May fix CPU starvation | GPU utilization improves; no RAM blowup |
| 3 | `batch_size=512, grad_accum=2` | Same effective batch, fewer steps | Fits <3.7 GB, val loss unchanged |
| 4 | `torch.compile` trial | Kernel fusion/graph capture | >=10% steady-state speed, no OOM |
| 5 | Muon hybrid | Faster convergence / possible memory reduction | Same val loss in fewer epochs or equal wall clock |
| 6 | AdEMAMix | Adam-like convergence improvement | Beats AdamW at equal wall clock |
| 7 | FFN x2 ablation | Direct compute cut | Keeps test loss within +0.02 |
| 8 | Optional cached windows | Eliminate collate bottleneck | Host-bound benchmark only |

## Bottom line

The first target should be not “Muon everywhere,” but a measured AdamW-local baseline that gets the hardware near saturation. After that, Muon is the most interesting research bet for AR-024 because it targets dense hidden matrices and could reduce epochs-to-quality; AdEMAMix is the safer optimizer-family bet; Sophia is promising in the literature but should be deferred until the cheaper changes have been exhausted. With the existing 5.65M-parameter/window-5 champion architecture, the 15-minute epoch target for 100,000 battles requires about a 2.3x improvement over the reported local 35-minute observation, and the repository already contains most of the runtime hooks needed to pursue it systematically.
