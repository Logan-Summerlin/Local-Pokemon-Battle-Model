# Local Pokemon Battle Model

Autonomous research stack for **Phase 4 imitation learning** on Gen 3 OU (ADV) Pokemon
battles, ported from Pokemon-Battle-AutoResearch and **retuned to train on a single local
NVIDIA GeForce GTX 1650 (4 GB)** — both pre-training and post-training — within an ~8-hour
budget, optionally driven autonomously by Claude Code inside Docker.

> **Hardware target:** GTX 1650 · 4 GB VRAM · ~15 GB RAM · mid-range CPU. The P8-Lean 50K reference run
> (63.21% top-1) was trained on this exact card in ~3.5 h, so the budgets are
> calibrated. The published champion **AR-020** (67.79% top-1) was trained on an A40; its
> literal `batch_size=1024` / `bf16` settings **do not transfer** — see below.

## What's in here

The full AutoResearch project: the `BattleTransformer` model, the replay→tensor data
pipeline, the experiment harness/registry, the governing `MASTER_RESEARCH_PLAN.md`, the
~100K committed processed battle tensors, and the committed P8-Lean reference checkpoint.

## GTX 1650 optimizations (the port's point)

| Lever | A40 champion | Local (this repo) | Where |
|---|---|---|---|
| Optimizer batch | `batch_size=1024` | micro-batch **256 × grad_accum 4** = eff. 1024 (~1.5 GB; model is activation-bound) | `gtx1650` profile / `--mode local` |
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

## Autonomous Claude Code sandbox (locked-down)

To let Claude Code run fully autonomously (`--dangerously-skip-permissions`) while
keeping it boxed in — **no host filesystem, no root, no Docker socket, no host
network, and internet access only to Anthropic + GitHub** — use the hardened
sandbox in **`docker/sandbox/`**. Full walkthrough:
**`docs/AUTONOMOUS_CLAUDE_SANDBOX_GUIDE.md`**.

It's two containers: a non-root `claude` container (CUDA + PyTorch + Node + Claude
Code) with **no direct internet**, behind a **Squid egress proxy** that allows only
an 11-host allow-list. `/workspace` is a Docker **named volume** (not a host bind
mount), capabilities are dropped, `no-new-privileges` is set, and RAM/CPU/PID are
capped. GPU is passed through via the NVIDIA runtime (no host/root access).

```bash
cd docker/sandbox
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi  # 0. confirm GPU reaches Docker
docker compose up -d --build                                               # 1. build + start (proxy + claude)
# 2. verify the network jail (see guide Section 2): Anthropic/GitHub reachable, everything else 403, direct egress 000
./copy-in.sh /path/to/Local-Pokemon-Battle-Model                           # 3. copy a project in (fixes ownership)
docker compose exec claude bash                                            # 4. log in once: `claude` -> "Log in with your Claude account"
#    then:  cd /workspace/<project> && pip install -e . --no-deps --no-build-isolation --user
#           claude --dangerously-skip-permissions
docker compose cp claude:/workspace/<project> ./review-output             # 5. copy results out and review the diff
```

Authentication uses your **Claude Pro/Max subscription** by default (no API key).
What it does **not** block: Claude can still read your login token, modify/delete
anything in `/workspace`, send file contents to Anthropic, and spend your quota —
so review the diff before copying changes back to your real repo. See the guide's
**Troubleshooting** section for the gotchas (Squid non-root logging,
`platform.claude.com` allow-listing, GPU passthrough, file ownership, the trust
prompt).

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/data/` | Replay parsing, observations, tensorization, dataset loading |
| `src/models/` | `BattleTransformer` architecture |
| `src/environment/` | Action space definition |
| `scripts/` | Training entry points; `run_curriculum_experiment_local.sh` for the GTX 1650 curriculum |
| `Autoresearch/` | Experiment harness, registry, **`MASTER_RESEARCH_PLAN.md`** |
| `docker/` | Dockerfile, compose, entrypoint for local autonomous training |
| `docker/sandbox/` | Hardened locked-down sandbox for autonomous Claude Code (proxy + non-root GPU container) |
| `docs/` | Setup guides: `LOCAL_DOCKER_AUTORESEARCH_SETUP_GUIDE.md`, the RunPod guide, and `AUTONOMOUS_CLAUDE_SANDBOX_GUIDE.md` |
| `data/`, `checkpoints/` | Committed processed battles, vocabs, P8-Lean checkpoint |
| `CLAUDE.md` | Autonomous agent operating manual (local) |

## Governance

Work follows `Autoresearch/MASTER_RESEARCH_PLAN.md` (currently **Phase A**) and the
invariants in `.claude/rules/memory.md` — move-identity + `shuffle_moves` always on,
hidden-information doctrine, rated-games-only, registry-first. Local compute constraints
are in MASTER_RESEARCH_PLAN.md §0.5.
