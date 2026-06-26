# Claude Code sandbox image. Node base (Claude Code is an npm package).
FROM node:20-bookworm-slim

# Tools Claude commonly needs inside the box. No sudo, no docker CLI on purpose.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl jq less ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally as root, then drop to a non-root user.
RUN npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force

# Non-root user that owns the workspace.
RUN useradd --create-home --shell /bin/bash claude \
    && mkdir -p /workspace /home/claude/.claude \
    && chown -R claude:claude /workspace /home/claude

# /workspace is backed by a named volume at runtime; it inherits this
# ownership the first time the volume is created.
VOLUME ["/workspace"]

USER claude
WORKDIR /workspace
