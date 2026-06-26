#!/usr/bin/env bash
# Container entrypoint: prepare the editable install + environment, then exec the
# requested command (an interactive shell by default, or `claude` for autonomy).
set -e

REPO=/workspace

# expandable_segments is also set in the image env, but re-assert in case the
# container was started with a different --env.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Register the local `src` package in editable mode against the mounted repo.
# Dependencies already live in the image, so --no-deps keeps this fast/offline.
if [ -f "${REPO}/pyproject.toml" ]; then
    pip install -e "${REPO}" --no-deps --quiet || \
        echo "WARN: editable install failed; is the repo mounted at ${REPO}?"
fi

# Quick GPU visibility report (non-fatal if no GPU / driver mismatch).
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "── GPU ──"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi
python - <<'PY' || true
import torch
print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)} | bf16_supported={torch.cuda.is_bf16_supported()}", flush=True)
PY

exec "$@"
