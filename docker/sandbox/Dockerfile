# Claude Code sandbox image WITH GPU training support.
#
# Mirrors the project's training image (docker/Dockerfile): CUDA 12.1.1 +
# PyTorch cu121 (Turing / GTX 1650 compatible) + the project's Python deps +
# Node/Claude Code. Adds a non-root 'claude' user and bakes ALL dependencies in
# so the running container needs NO PyPI access (the egress allow-list blocks it).
#
# NOTE: `docker build` runs on the host with full internet, so pulling the CUDA
# base image and the torch/pip wheels here is fine. Only the *running* container
# is network-restricted to the proxy allow-list.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Reduce fragmentation OOMs on the 4 GB card (see scripts/train_phase4.py).
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Keep HuggingFace cache on the workspace volume.
    HF_HOME=/workspace/.hf_cache

# ── System packages: Python 3.11, git, Node.js (for Claude Code), CLI tools ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl ca-certificates build-essential \
        jq less ripgrep \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# setuptools/wheel are needed so the OFFLINE editable install at runtime can use
# --no-build-isolation (no PyPI fetch). pip upgraded for good measure.
RUN python -m pip install --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.1 build — Turing-compatible), pinned to the project's window ──
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1

# ── Project runtime + dev dependencies (mirrors pyproject.toml) ──
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

# ── Claude Code CLI ──
RUN npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force

# ── Non-root user that owns the workspace ──
RUN useradd --create-home --shell /bin/bash claude \
    && mkdir -p /workspace /home/claude/.claude \
    && chown -R claude:claude /workspace /home/claude

VOLUME ["/workspace"]

USER claude
WORKDIR /workspace
