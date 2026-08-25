#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

DATASET_DIR="${MEMGUI3K_DIR:-$REPO_ROOT/data/MemGUI-3K}"
TRAINING_DATA_DIR="${MEMGUI3K_TRAINING_DATA_DIR:-$DATASET_DIR/training_data}"
INPUT="${MEMGUI_EVAL_INPUT:-$TRAINING_DATA_DIR/test_sft.jsonl}"
GOLD="${MEMGUI_EVAL_GOLD:-$INPUT}"

API_BASE="${OPENAI_BASE_URL:-${QWEN_BASE_URL:-http://localhost:8000/v1}}"
API_KEY="${OPENAI_API_KEY:-${QWEN_API_KEY:-EMPTY}}"
MODEL_SPECS="${MEMGUI_EVAL_MODELS:-MemGUI-8B-SFT=MemGUI-8B-SFT}"

WORKERS=8
RUN_ID=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --input)
      INPUT="$2"
      GOLD="$2"
      shift 2
      ;;
    --gold)
      GOLD="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$INPUT" ]]; then
  echo "Missing evaluation input: $INPUT" >&2
  echo "Build test_sft.jsonl first with: bash scripts/build_memgui3k_training_data.sh" >&2
  exit 2
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

RESULTS_DIR="$SCRIPT_DIR/results/$RUN_ID"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  MemGUI-3K Offline Evaluation"
echo "============================================================"
echo "  Run ID:    $RUN_ID"
echo "  API Base:  $API_BASE"
echo "  Input:     $INPUT"
echo "  Gold:      $GOLD"
echo "  Models:    $MODEL_SPECS"
echo "  Workers:   $WORKERS"
echo "  Results:   $RESULTS_DIR"
echo ""

cat > "$RESULTS_DIR/run_config.json" <<EOF
{
  "run_id": "$RUN_ID",
  "timestamp": "$(date -Iseconds)",
  "api_base": "$API_BASE",
  "input": "$INPUT",
  "gold": "$GOLD",
  "workers": $WORKERS,
  "model_specs": "$MODEL_SPECS"
}
EOF

IFS=',' read -r -a SPECS <<< "$MODEL_SPECS"
for SPEC in "${SPECS[@]}"; do
  MODEL_NAME="${SPEC%%=*}"
  MODEL_ID="${SPEC#*=}"

  if [[ -z "$MODEL_NAME" || -z "$MODEL_ID" || "$MODEL_NAME" == "$MODEL_ID" && "$SPEC" != *"="* ]]; then
    echo "Invalid model spec: $SPEC" >&2
    echo "Use MEMGUI_EVAL_MODELS='NameA=model-a,NameB=model-b'" >&2
    exit 2
  fi

  SAFE_MODEL_NAME="$(echo "$MODEL_NAME" | tr '/ :' '___')"
  PRED_FILE="$RESULTS_DIR/${SAFE_MODEL_NAME}_predictions.jsonl"
  REPORT_FILE="$RESULTS_DIR/${SAFE_MODEL_NAME}_report.json"

  echo "------------------------------------------------------------"
  echo "  Model: $MODEL_NAME ($MODEL_ID)"
  echo "------------------------------------------------------------"

  python run_offline_eval.py \
    --input "$INPUT" \
    --output "$PRED_FILE" \
    --api-base "$API_BASE" \
    --api-model "$MODEL_ID" \
    --api-key "$API_KEY" \
    --workers "$WORKERS" \
    --resume \
    "${EXTRA_ARGS[@]}"

  python evaluate_predictions.py \
    --predictions "$PRED_FILE" \
    --gold "$GOLD" \
    --output "$REPORT_FILE"
done

python compare_results.py \
  --results-dir "$RESULTS_DIR" \
  --output "$RESULTS_DIR/comparison_summary.md"

echo "Done: $RESULTS_DIR"
