# Pokemon Battle AutoResearch — local autonomous training image.
#
# Targets a single NVIDIA GTX 1650 (4 GB, Turing / compute capability 7.5).
# CUDA 12.1 runtime + PyTorch cu121 wheels support Turing. The image carries ONLY
# the Python/Node environment; the repository (code + the ~400 MB processed data +
# checkpoints) is bind-mounted at runtime so commits/pushes land on the host.
#
# Build context is kept tiny by .dockerignore (data/, checkpoints/, .git excluded).

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Reduce fragmentation OOMs on the 4 GB card (see scripts/train_phase4.py).
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Keep HuggingFace dataset/model downloads on a mounted cache volume.
    HF_HOME=/workspace/.hf_cache

# ── System packages: Python 3.11, git, curl, Node.js (for Claude Code) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl ca-certificates build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# ── PyTorch (CUDA 12.1 build — Turing-compatible) ──
# Pinned inside the project's torch>=2.4,<2.7 window.
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1

# ── Project runtime + dev dependencies ──
# Listed explicitly (mirrors pyproject.toml) so the build context stays tiny and
# does not need the source tree. The editable install of the local `src` package
# itself happens at container start (entrypoint), against the mounted repo.
RUN pip install \
        "numpy>=1.26.0,<2.0" \
        "lz4>=4.3.0" \
        "huggingface_hub>=0.20.0" \
        "tqdm>=4.66.0" \
        "rich>=13.7.0" \
        "pyyaml>=6.0" \
        "pytest>=8.0.0,<8.1" \
        "pytest-cov>=4.1.0,<4.2" \
        "ruff>=0.3.0,<0.4"

# ── Claude Code CLI (the autonomous research agent) ──
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /workspace

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
