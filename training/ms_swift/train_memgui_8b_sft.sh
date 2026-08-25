#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-$REPO_ROOT/.cache/tmp}"

mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/data/MemGUI-3K/training_data/train_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/memgui-sft-qwen3vl}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-memgui-sft-qwen3vl}"

if ! command -v swift >/dev/null 2>&1; then
  echo "swift command not found. Install ms-swift before training." >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "Missing training file: $TRAIN_FILE" >&2
  echo "Build training data first with: bash scripts/build_memgui3k_training_data.sh" >&2
  exit 2
fi

swift sft \
  --model "$MODEL" \
  --train_type lora \
  --dataset "$TRAIN_FILE" \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 8 \
  --eval_steps 1 \
  --save_steps 1 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --max_length "$MAX_LENGTH" \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --report_to "$REPORT_TO" \
  --run_name "$RUN_NAME" \
  "$@"
