# Release Checklist

This checklist keeps the Hugging Face dataset repository and the GitHub project
repository cleanly separated.

## GitHub: memgui-agent-anonymous/MemGUI-Agent

The GitHub repository should host:

- MemGUI-3K data download, image restoration, packaging, and training-data
  conversion scripts.
- MemGUI-8B-SFT training templates.
- MemGUI-3K offline evaluation code.
- Project-page assets and paper-facing documentation.

## Hugging Face Dataset: memgui-agent-anonymous/MemGUI-3K

The dataset repository should host only:

- Dataset card.
- `metadata.json`.
- `split.json` for Dataset Viewer.
- `system_prompt.txt`.
- `train_trajectories.jsonl`.
- `test_trajectories.jsonl`.
- `image_archives/images.z*` and `image_archives/images.zip`.

Do not place processing scripts or long usage guides in the dataset repository.
Point users to:

https://github.com/memgui-agent-anonymous/MemGUI-Agent

## Hugging Face Model: memgui-agent-anonymous/MemGUI-8B-SFT

The model repository should host:

- Model weights and tokenizer/preprocessor files.
- A model card with links to the paper, dataset, and GitHub repository.
- Minimal inference examples.

Training scripts and evaluation scripts should remain in the GitHub repository.
