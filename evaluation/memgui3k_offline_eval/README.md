# MemGUI-3K Offline Evaluation

This toolkit evaluates model outputs on the MemGUI-3K test split without
running live Android rollouts. It feeds each test step's gold system prompt,
user prompt, and screenshot into a model, then compares the generated response
against the gold assistant response.

## Inputs

Build the test SFT file first:

```bash
bash scripts/build_memgui3k_training_data.sh
```

Default input:

```text
data/MemGUI-3K/training_data/test_sft.jsonl
```

## Run One Model

```bash
cd evaluation/memgui3k_offline_eval

OPENAI_BASE_URL=http://localhost:8000/v1 \
OPENAI_API_KEY=EMPTY \
python run_offline_eval.py \
  --input ../../data/MemGUI-3K/training_data/test_sft.jsonl \
  --output results/memgui8b_predictions.jsonl \
  --api-model MemGUI-8B-SFT \
  --workers 8 \
  --resume
```

Then evaluate:

```bash
python evaluate_predictions.py \
  --predictions results/memgui8b_predictions.jsonl \
  --gold ../../data/MemGUI-3K/training_data/test_sft.jsonl \
  --output results/memgui8b_report.json
```

## Run Multiple Models

`run_all.sh` reads OpenAI-compatible API settings from environment variables:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
export MEMGUI_EVAL_MODELS='MemGUI-8B-SFT=MemGUI-8B-SFT,Qwen3-VL-8B=Qwen/Qwen3-VL-8B-Instruct'

bash evaluation/memgui3k_offline_eval/run_all.sh --workers 8
```

Each model spec has the form `display_name=api_model_id`.

## Metrics

The evaluator reports:

- Action type accuracy and action match success.
- Memory action trigger precision, recall, and F1.
- Folding presence, range accuracy, and deep/shallow folding breakdowns.
- Format compliance for the ConAct response fields.

Results are written under:

```text
evaluation/memgui3k_offline_eval/results/<run_id>/
```
