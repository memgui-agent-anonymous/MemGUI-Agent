#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${MEMGUI3K_DIR:-data/MemGUI-3K}"
if [[ $# -ge 1 ]]; then
  DATASET_DIR="$1"
fi

ARCHIVE_DIR="$DATASET_DIR/image_archives"
SPLIT_ZIP="$ARCHIVE_DIR/images.zip"
FULL_ZIP="$ARCHIVE_DIR/images-full.zip"

if [[ ! -d "$ARCHIVE_DIR" ]]; then
  echo "Missing image archive directory: $ARCHIVE_DIR" >&2
  exit 2
fi
if [[ ! -f "$SPLIT_ZIP" ]]; then
  echo "Missing split zip main file: $SPLIT_ZIP" >&2
  exit 2
fi

echo "Dataset directory: $DATASET_DIR"
echo "Archive directory: $ARCHIVE_DIR"

rm -f "$FULL_ZIP"

echo "Merging split zip archive..."
if ! (cd "$ARCHIVE_DIR" && zip -s 0 images.zip --out images-full.zip); then
  echo "zip -s 0 failed; trying zip -FF repair mode..."
  (cd "$ARCHIVE_DIR" && zip -FF images.zip --out images-full.zip)
fi

echo "Unzipping screenshots into dataset root..."
unzip -q -o "$FULL_ZIP" -d "$DATASET_DIR"

if [[ ! -d "$DATASET_DIR/images" ]]; then
  echo "Restore failed: $DATASET_DIR/images was not created." >&2
  exit 3
fi

N_IMAGES="$(find "$DATASET_DIR/images" -type f -name '*.png' | wc -l)"
echo "Restored PNG files: $N_IMAGES"
echo "Done."
