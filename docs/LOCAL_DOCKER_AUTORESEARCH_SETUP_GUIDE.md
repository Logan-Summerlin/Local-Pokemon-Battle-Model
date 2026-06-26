# Local Docker + Autonomous Claude Code Setup (GTX 1650)

This guide brings the AutoResearch stack onto a **local workstation** and runs Claude
Code autonomously inside Docker to train the BattleTransformer on a single
**NVIDIA GeForce GTX 1650 (4 GB)**.

> Target machine (the box this was tuned for): **GTX 1650, 4 GB VRAM · ~15 GB system
> RAM · mid-range CPU.** The frozen anchor (P8-Lean 50K) was in fact trained on this
> exact card in ~3.5 h, so the budgets below are calibrated, not guessed.

---

## 0. Why a special profile for this GPU?

The published champion (**AR-041**) was trained on an A40 (44 GB) with `batch_size=1024`
and `amp=bf16`. Neither choice transfers to a GTX 1650:

| Issue | A40 setting | GTX 1650 reality | Local fix |
|---|---|---|---|
| **VRAM** | batch 1024 fits in 44 GB | 4 GB OOMs almost immediately | micro-batch **64 × grad_accum 16 = effective 1024** |
| **Precision** | `bf16` (Ampere has bf16 cores) | Turing has **no bf16 hardware**; bf16 autocast upcasts to fp32 → no speedup | `amp=fp16` — Turing runs **2× packed FP16** on its CUDA cores (faster *and* halves activation memory) |
| **Throughput** | ~thousands ex/s | ~10–40× slower | reduce data (25K battles) + epochs to stay under ~8 h |
| **Fragmentation** | irrelevant | small heap fragments → spurious OOM | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

These are baked into:
- `scripts/train_phase4.py --mode local`
- the **`gtx1650`** (alias `local`) profile in `Autoresearch/run_experiment.py`
- `scripts/run_curriculum_experiment_local.sh`
- the Docker image (`PYTORCH_CUDA_ALLOC_CONF`, `OMP_NUM_THREADS`, `mem_limit`).

---

## 1. Host prerequisites (one-time)

1. **NVIDIA driver** ≥ 530 (needed for the CUDA 12.1 runtime). Verify:
   ```bash
   nvidia-smi      # should list "NVIDIA GeForce GTX 1650" and a driver version
   ```
2. **Docker Engine** (Linux) or Docker Desktop with WSL2 GPU support (Windows).
3. **NVIDIA Container Toolkit** — this is what lets containers see the GPU:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
   Smoke-test GPU passthrough:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```
   > **Windows:** install the NVIDIA driver on Windows (not inside WSL), run Docker
   > Desktop on the WSL2 backend, and use this same toolkit inside your WSL2 distro.
   > Training inside Linux/WSL also gives you `fork`-based DataLoader workers, which
   > are dramatically faster than native-Windows `spawn` workers.

---

## 2. Clone the repo on the host

```bash
git clone https://github.com/logan-summerlin/local-pokemon-battle-model.git
cd local-pokemon-battle-model
```

The ~100K processed battle tensors (`data/processed/battles/`), vocabs, and the anchor
checkpoint are committed, so you can train immediately with **no download**.

---

## 3. Build the image

```bash
docker compose -f docker/docker-compose.yml build
```

The build installs Python 3.11, PyTorch (cu121), the project deps, Node.js, and the
Claude Code CLI. The repo itself is **bind-mounted at runtime** (`..:/workspace`), so the
image stays small and every commit/push you make inside the container lands on your host
clone.

---

## 4. Sanity check the environment

```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  bash -lc 'pytest tests/test_transformer.py -q && \
            python scripts/train_phase4.py --mode smoke'
```

The entrypoint prints `cuda_available=True`, the device name, and
`bf16_supported=False` (expected on Turing — confirming fp16 is the right choice).

---

## 5. Pre-training (imitation learning)

### Option A — out-of-the-box, single stage (recommended first run, no download)

Trains the champion-class architecture on the committed 25K battles:

```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  python Autoresearch/run_experiment.py \
    --name local_pretrain_v1 --parent anchor --tier 2 \
    --profile gtx1650 --budget-minutes 420 \
    --config-override epochs=12 cosine_epochs=24 \
    --hypothesis "Local GTX1650 champion-class pretrain on 25K battles"
```

Or the bare convenience preset (no registry entry):
```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  python scripts/train_phase4.py --mode local \
    --split-head --move-identity --shuffle-moves --no-value-head
```

### Option B — full 2-stage Elo curriculum (matches the champion procedure)

First build the Elo-stratified manifests (downloads the Metamon tar once, ~GBs):
```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  python scripts/prepare_curriculum_data.py
```
Then run the GTX-1650 curriculum (stage 1 → resume → stage 2):
```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  bash scripts/run_curriculum_experiment_local.sh baseline
```

---

## 6. Post-training (fine-tuning)

Post-training reuses the same `--resume-from` path on a smaller, higher-quality slice
(Elo stage 2, hard-example mining, or Phase D synthetic data once it exists). Example —
fine-tune a pretrained checkpoint for a few epochs at a lower LR:

```bash
docker compose -f docker/docker-compose.yml run --rm trainer \
  python Autoresearch/run_experiment.py \
    --name local_finetune_v1 \
    --parent AR-XYZ \
    --tier 2 --profile gtx1650 --budget-minutes 120 \
    --config-override \
        epochs=3 cosine_epochs=6 lr=1e-4 warmup_steps=50 \
        resume_from=checkpoints/autoresearch_ar-xyz_local_pretrain_v1
```

Keep a **hard regression guard**: if real-replay validation accuracy drops, reduce the
fine-tune LR/epochs or the synthetic mix (see MASTER_RESEARCH_PLAN.md §6).

---

## 7. The ~8-hour budget — calibrate, don't guess

Throughput on *your* card is the source of truth. Every run writes
`examples_per_sec` and wall time into its report. To calibrate:

```bash
# One short timing run, then read the numbers back:
docker compose -f docker/docker-compose.yml run --rm trainer \
  python scripts/train_phase4.py --mode local --epochs 1 \
    --split-head --move-identity --shuffle-moves --no-value-head
python - <<'PY'
import json; r=json.load(open('checkpoints/phase4_local/training_report.json'))
em=r['epoch_metrics'][0]
print('examples/sec:', em['examples_per_sec'], '| epoch sec:', em['epoch_time_sec'])
print('train examples:', r['data_stats']['train_examples'])
PY
```

Budgeting rule of thumb (anchor baseline: ~795 ex/s at window 2 / 1.7 M params / bf16):
- **time ≈ train_examples × epochs ÷ examples_per_sec.**
- If a planned run exceeds your budget, dial down in this order: `num_battles` →
  `epochs` → `max_window` (5→3) → `hidden_dim`/`num_layers`. Each is a single `--config-override`.

A reasonable end-to-end local target: **pretrain ≤ ~6 h + post-train ≤ ~1–2 h ≤ 8 h.**

---

## 8. Autonomous Claude Code

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # on the host, before launching
docker compose -f docker/docker-compose.yml run --rm trainer claude
```

Then paste:

```
Read CLAUDE.md. This is an autonomous local research session on a GTX 1650 (4 GB).
Follow Autoresearch/MASTER_RESEARCH_PLAN.md — start at Phase A. Use the `gtx1650`
profile and an explicit --budget-minutes for every run. Verify the anchor, then work
the priority queue. Commit after each experiment and push to the working branch.
Do not exceed ~8 h of training per run. Go.
```

For long unattended sessions, run it inside `tmux`/`screen` (or
`docker compose ... run -d`) so it survives disconnects. Claude Code uses
sleep/wake cycles while a training subprocess runs and records every outcome in
`Autoresearch/experiment_registry.json`.

> Autonomy guardrails live in `CLAUDE.md` and `.claude/settings.json` (git, training,
> and eval commands are pre-allowed; destructive `rm -rf` and edits to the frozen
> data/observation layer are denied).

---

## 9. Monitoring

From a second host terminal (training writes through the bind mount):
```bash
watch -n 30 nvidia-smi                              # GPU/VRAM
python Autoresearch/leaderboard.py                  # ranked experiments
tail -f checkpoints/phase4_local/training_report.json 2>/dev/null
git -C . log --oneline -20                          # experiment commits
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Lower `batch_size` (64→48→32); grad_accum keeps the effective batch. Confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Drop `max_window` 5→3. |
| Training pinned to CPU | `cuda_available=False` → host driver/toolkit issue; re-run the §1 GPU smoke test. |
| Very slow epochs | Ensure `amp=fp16` (not bf16/off); raise `num_workers` toward your physical core count; confirm you are on Linux/WSL (not native-Windows spawn). |
| Desktop freezes / host OOM | The 4 GB card shares nothing with RAM, but the in-RAM dataset + workers can; lower `num_battles` and keep compose `mem_limit`/`shm_size`. |
| Run killed at the tier cap | Pass a larger `--budget-minutes`; tier caps assume an A40. |

---

## 11. Stopping & resuming

- **Graceful:** `Ctrl+C` in the Claude session; the current experiment subprocess is
  terminated and the registry/checkpoints are preserved.
- **Resume:** relaunch `claude` with the same prompt — it reads the registry and
  continues. Per-experiment, use `--config-override resume_from=<checkpoint_dir>`.

All progress lives in git commits, `Autoresearch/experiment_registry.json`, and the
checkpoints under `checkpoints/` — all on your host via the bind mount.
