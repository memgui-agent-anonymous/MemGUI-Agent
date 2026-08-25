#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_DIR="${MEMGUI3K_DIR:-$REPO_ROOT/data/MemGUI-3K}"
OUTPUT_DIR="${MEMGUI3K_TRAINING_DATA_DIR:-$DATASET_DIR/training_data}"

if [[ ! -f "$DATASET_DIR/train_trajectories.jsonl" ]]; then
  echo "Missing dataset files under: $DATASET_DIR" >&2
  echo "Download MemGUI-3K first with: bash scripts/download_memgui3k.sh" >&2
  exit 2
fi
if [[ ! -d "$DATASET_DIR/images" ]]; then
  echo "Missing restored images directory: $DATASET_DIR/images" >&2
  echo "Restore screenshots first with: bash scripts/restore_memgui3k_images.sh" >&2
  exit 2
fi

python3 "$REPO_ROOT/data/memgui3k/training_data_converter/build_training_data.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
