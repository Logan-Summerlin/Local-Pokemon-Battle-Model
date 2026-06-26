# RunPod AutoResearch Setup Guide

## Recommended Instance

- **GPU:** A40 48GB or similar.
- **CPU:** 8+ vCPUs.
- **RAM:** 32GB+.
- **Disk:** 100GB persistent volume.
- **Base image:** Python 3.11 with CUDA-enabled PyTorch.

## 1. Clone and install

```bash
cd /workspace
git clone <your-repo-url> Pokemon-Battle-Autoresearch
cd Pokemon-Battle-Autoresearch

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. Download raw replay data

### General sampler
```bash
python scripts/download_replays.py --generation gen3ou --sample-size 10000 --elo-threshold 1300 --output-dir data/raw
```

### Stratified Gen 3 OU sampler
```bash
python scripts/download_replays_stratified.py --target-size 100000 --output-dir data/raw
```

## 3. Process replays

```bash
python scripts/process_dataset.py --input-dir data/raw --output-dir data/processed --generation gen3ou
```

## 4. Validate the stack

```bash
pytest tests/test_parser.py tests/test_observation.py tests/test_tensorizer.py tests/test_transformer.py tests/test_base_stats.py
python scripts/train_p8_lean.py --dry-run --num-battles 10 --seeds 42
```

## 5. Launch training (manual mode)

### P8-Lean
```bash
python scripts/train_p8_lean.py --num-battles 10000 --seeds 42 --batch-size 64
```

### P4
```bash
python scripts/train_p4_25k.py --num-battles 25000 --seeds 42 --batch-size 32
```

## 6. Run experiment management (manual mode)

```bash
python Autoresearch/run_experiment.py \
  --name window5 \
  --parent anchor \
  --tier 1 \
  --config-override max_window=5

python Autoresearch/leaderboard.py
```

---

## 7. Install Claude Code (for autonomous research)

Claude Code is the autonomous research agent. Install it on the RunPod instance:

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Verify installation
claude --version
```

If npm is not available, install Node.js first:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
```

## 8. Configure for autonomous operation

The repository comes pre-configured for autonomous Claude Code operation:
- `.claude/settings.json` — permissions for git operations, training, evaluation
- `.claude/rules/memory.md` — project memory and autonomous operation directives
- `CLAUDE.md` — full experiment protocol, priority queue, NEVER STOP directive

Verify the configuration:
```bash
cat .claude/settings.json   # Check permissions include git reset, git checkout
cat CLAUDE.md               # Should contain NEVER STOP directive and priority queue
```

## 9. Launch autonomous research

Navigate to the repository and launch Claude Code with the autonomy prompt:

```bash
cd /workspace/Pokemon-Battle-Autoresearch
claude
```

Then paste this prompt:

```
Read CLAUDE.md. This is a fully autonomous research session.
Set up the experiment branch, verify the anchor, and begin the experiment loop.
Run experiments continuously. Do not stop or ask for permission.
Target: beat the 63.21% Top-1 anchor accuracy.
Go.
```

Claude Code will:
1. Read the project configuration
2. Verify the anchor checkpoint
3. Begin running experiments from the priority queue
4. Use 5-minute sleep/wake cycles during training runs
5. Commit progress, revert failures, advance the branch on successes
6. Continue indefinitely until interrupted

## 10. Monitoring progress

While Claude Code is running, you can monitor from a separate terminal:

```bash
# Check experiment registry
cat Autoresearch/experiment_registry.json | python -m json.tool

# View leaderboard
python Autoresearch/leaderboard.py

# Check git log for experiment commits
git log --oneline -20

# View latest experiment notes
ls -lt Autoresearch/notes/

# Check GPU utilization
nvidia-smi
```

## 11. Stopping

- **Graceful stop:** Press `Ctrl+C` in the Claude Code terminal. Current experiment may complete first.
- **Force stop:** Close the terminal session. Training subprocess will be killed.
- **Resume later:** Re-launch Claude Code with the same prompt. It will read the registry and continue from where it left off.

All progress is preserved in git commits, the experiment registry, and experiment notes.
