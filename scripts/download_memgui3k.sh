#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${MEMGUI3K_REPO_ID:-memgui-agent-anonymous/MemGUI-3K}"
LOCAL_DIR="${MEMGUI3K_DIR:-data/MemGUI-3K}"

if [[ $# -ge 1 ]]; then
  REPO_ID="$1"
fi
if [[ $# -ge 2 ]]; then
  LOCAL_DIR="$2"
fi

echo "Downloading dataset: $REPO_ID"
echo "Local directory:     $LOCAL_DIR"

hf download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$LOCAL_DIR"

echo "Done: $LOCAL_DIR"
