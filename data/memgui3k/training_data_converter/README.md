# MemGUI-3K Training Data Converter

This folder rebuilds step-level multimodal training JSONL files from the
released MemGUI-3K trajectory data.

It does not require the original rollout directories. The input is the released
MemGUI-3K layout:

```text
MemGUI-3K/
|-- system_prompt.txt
|-- train_trajectories.jsonl
|-- test_trajectories.jsonl
`-- images/
```

## Output

The converter writes:

```text
training_data/
|-- train_sft.jsonl
`-- test_sft.jsonl
```

Each line is one evaluator-approved reasonable step:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": ["data:image/png;base64,..."],
  "task_id": "061-S4689",
  "step": 5
}
```

The format is compatible with ms-swift multimodal supervised training and can
also be adapted for other training or reinforcement-learning pipelines.

## Usage

From the MemGUI-Agent repository root:

```bash
python3 data/memgui3k/training_data_converter/build_training_data.py \
  --dataset-dir data/MemGUI-3K \
  --output-dir data/MemGUI-3K/training_data
```

You can also use the convenience wrapper:

```bash
bash scripts/build_memgui3k_training_data.sh
```

Quick smoke test:

```bash
python3 data/memgui3k/training_data_converter/build_training_data.py \
  --dataset-dir data/MemGUI-3K \
  --max-trajectories 2 \
  --output-dir /tmp/memgui3k_training_smoke
```

## Notes

- The converter keeps only steps where `is_reasonable` is true.
- Images are read from `images/` and embedded as base64 PNG data URLs.
- The shared system prompt is read from `system_prompt.txt`.
- Existing output files are overwritten unless `--no-overwrite` is passed.
