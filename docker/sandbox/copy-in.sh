#!/usr/bin/env bash
# Copy a project INTO the sandbox /workspace volume with correct (non-root)
# ownership so Claude Code can edit it.
#
# Why this script exists: `docker compose cp` writes files into the volume as
# root, but the sandbox runs as the non-root 'claude' user. We can't chown
# inside the running container because it has `cap_drop: ALL` (no CAP_CHOWN).
# So we fix ownership with a THROWAWAY root container that has default caps,
# WITHOUT weakening the running sandbox (it keeps cap_drop: ALL + user: claude).
#
# Usage (run from docker/sandbox/):
#   ./copy-in.sh <path-to-project> [dest-name]
set -euo pipefail

SRC="${1:?Usage: ./copy-in.sh <path-to-project> [dest-name]}"
NAME="${2:-$(basename "$SRC")}"

# Compose project name defaults to this directory's name ("sandbox");
# the volume and image names follow from it.
PROJECT="$(basename "$PWD")"
VOLUME="${PROJECT}_workspace"
IMAGE="${PROJECT}-claude"

echo ">> Copying $SRC -> /workspace/$NAME ..."
docker compose cp "$SRC" "claude:/workspace/$NAME"

echo ">> Fixing ownership (throwaway root container; sandbox stays locked) ..."
docker run --rm -u 0:0 -v "${VOLUME}:/workspace" "$IMAGE" \
    chown -R claude:claude "/workspace/$NAME"

echo ">> Done. /workspace/$NAME is owned by the non-root 'claude' user."
