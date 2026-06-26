# Local Pokemon Battle Model

Autonomous research stack for **Phase 4 imitation learning** on Gen 3 OU (ADV) Pokemon
battles, ported from Pokemon-Battle-AutoResearch and **retuned to train on a single local
NVIDIA GeForce GTX 1650 (4 GB)** — both pre-training and post-training — within an ~8-hour
budget, optionally driven autonomously by Claude Code inside Docker.

> **Hardware target:** GTX 1650 · 4 GB VRAM · ~15 GB RAM · mid-range CPU. The frozen anchor
> (P8-Lean 50K, 63.21% top-1) was trained on this exact card in ~3.5 h, so the budgets are
> calibrated. The published champion **AR-041** (67.79% top-1) was trained on an A40; its
> literal `batch_size=1024` / `bf16` settings **do not transfer** — see below.

## What's in here

The full AutoResearch project: the `BattleTransformer` model, the replay→tensor data
pipeline, the experiment harness/registry, the governing `MASTER_RESEARCH_PLAN.md`, the
~100K committed processed battle tensors, and the frozen anchor checkpoint.

## GTX 1650 optimizations (the port's point)

| Lever | A40 champion | Local (this repo) | Where |
|---|---|---|---|
| Optimizer batch | `batch_size=1024` | micro-batch **64 × grad_accum 16** = eff. 1024 (fits 4 GB) | `gtx1650` profile / `--mode local` |
| Precision | `bf16` | **`fp16`** (Turing: 2× packed FP16, no bf16 HW) | profile / `train_phase4.py` |
| Allocator | n/a | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `train_phase4.py` / Docker |
| Data scale | 50–100K battles | 25K (in-RAM dataset ≈ 2–3 GB) | profile |
| Wall clock | tier caps (A40) | explicit `--budget-minutes`, ≤ ~8 h | docs / curriculum script |

Encoded in three places: the **`gtx1650`** (alias `local`) profile in
`Autoresearch/run_experiment.py`, the **`--mode local`** preset in
`scripts/train_phase4.py`, and `scripts/run_curriculum_experiment_local.sh`.

## Quick start (Docker, recommended)

See **`docs/LOCAL_DOCKER_AUTORESEARCH_SETUP_GUIDE.md`** for the full guide (host driver +
NVIDIA Container Toolkit, build, autonomous Claude Code, monitoring, troubleshooting).

```bash
# Host prereqs: NVIDIA driver + Docker + nvidia-container-toolkit (see the guide).
docker compose -f docker/docker-compose.yml build

# Pre-train (champion-class, 25K battles, ~fits 8 h):
docker compose -f docker/docker-compose.yml run --rm trainer \
  python Autoresearch/run_experiment.py --name local_pretrain_v1 --parent anchor \
    --tier 2 --profile gtx1650 --budget-minutes 420 \
    --config-override epochs=12 cosine_epochs=24

# Autonomous research agent:
export ANTHROPIC_API_KEY=sk-ant-...
docker compose -f docker/docker-compose.yml run --rm trainer claude
```

## Quick start (bare metal)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # install a cu121 torch build for GPU
pytest                            # validate the stack

# One-command local baseline run on the committed data:
python scripts/train_phase4.py --mode local \
  --split-head --move-identity --shuffle-moves --no-value-head
```

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/data/` | Replay parsing, observations, tensorization, dataset loading |
| `src/models/` | `BattleTransformer` architecture |
| `src/environment/` | Action space definition |
| `scripts/` | Training entry points; `run_curriculum_experiment_local.sh` for the GTX 1650 curriculum |
| `Autoresearch/` | Experiment harness, registry, **`MASTER_RESEARCH_PLAN.md`** |
| `docker/` | Dockerfile, compose, entrypoint for local autonomous training |
| `docs/` | `LOCAL_DOCKER_AUTORESEARCH_SETUP_GUIDE.md` (local) and the RunPod guide (cloud) |
| `data/`, `checkpoints/` | Committed processed battles, vocabs, anchor checkpoint |
| `CLAUDE.md` | Autonomous agent operating manual (local) |

## Governance

Work follows `Autoresearch/MASTER_RESEARCH_PLAN.md` (currently **Phase A**) and the
invariants in `.claude/rules/memory.md` — move-identity + `shuffle_moves` always on,
hidden-information doctrine, rated-games-only, registry-first. Local compute constraints
are in MASTER_RESEARCH_PLAN.md §0.5.
