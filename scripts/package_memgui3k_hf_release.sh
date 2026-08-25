#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/package_memgui3k_hf_release.sh <source-memgui3k-dir> <output-hf-release-dir>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC_DIR="$(realpath "$1")"
DST_DIR="$(realpath -m "$2")"
SPLIT_SIZE="${MEMGUI3K_ZIP_SPLIT_SIZE:-10g}"
README_TEMPLATE="$REPO_ROOT/data/memgui3k/hf_dataset_card/README.md"

for path in \
  "$SRC_DIR/metadata.json" \
  "$SRC_DIR/split.json" \
  "$SRC_DIR/system_prompt.txt" \
  "$SRC_DIR/train_trajectories.jsonl" \
  "$SRC_DIR/test_trajectories.jsonl" \
  "$SRC_DIR/images"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required source path: $path" >&2
    exit 2
  fi
done

mkdir -p "$DST_DIR/image_archives"

cp "$README_TEMPLATE" "$DST_DIR/README.md"
cp "$SRC_DIR/metadata.json" "$DST_DIR/metadata.json"
cp "$SRC_DIR/split.json" "$DST_DIR/split.json"
cp "$SRC_DIR/system_prompt.txt" "$DST_DIR/system_prompt.txt"
cp "$SRC_DIR/train_trajectories.jsonl" "$DST_DIR/train_trajectories.jsonl"
cp "$SRC_DIR/test_trajectories.jsonl" "$DST_DIR/test_trajectories.jsonl"

python3 - "$DST_DIR/metadata.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = json.loads(path.read_text(encoding="utf-8"))

meta["image_packaging"] = {
    "path": "image_archives/",
    "format": "split zip",
    "split_size": "10g",
    "archive_entry_root": "images/",
    "usage_docs": "https://github.com/memgui-agent-anonymous/MemGUI-Agent",
}

top_level = []
for item in meta.get("top_level_files", []):
    if item.get("path") in {"training_data_converter/", "images/", "image_archives/"}:
        continue
    top_level.append(item)
top_level.append(
    {
        "path": "image_archives/",
        "description": "10GB split zip archives containing the released screenshot images.",
    }
)
meta["top_level_files"] = top_level

notes = [
    note for note in meta.get("format_notes", [])
    if "training_data_converter" not in note and "image_archives" not in note and "split zip" not in note
]
notes.append(
    "The Hugging Face release stores screenshots in image_archives/ as 10GB split zip files; usage scripts live in https://github.com/memgui-agent-anonymous/MemGUI-Agent."
)
meta["format_notes"] = notes

path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

rm -f "$DST_DIR"/image_archives/images.z* "$DST_DIR"/image_archives/images.zip

echo "Creating split zip archive from $SRC_DIR/images"
echo "Output: $DST_DIR/image_archives/images.zip"
echo "Split size: $SPLIT_SIZE"

(cd "$SRC_DIR" && zip -0 -q -r -s "$SPLIT_SIZE" "$DST_DIR/image_archives/images.zip" images)

N_IMAGES="$(zip -sf "$DST_DIR/image_archives/images.zip" | awk '/^  images\\/.*\\.png$/ {count++} END {print count+0}')"
echo "Archived PNG files: $N_IMAGES"
echo "Prepared HF release directory: $DST_DIR"
