# MemGUI-8B-SFT Training

MemGUI-8B-SFT is trained with
[ms-swift](https://github.com/modelscope/ms-swift) from the MemGUI-3K
step-level multimodal training JSONL files.

## Prepare Data

From the repository root:

```bash
bash scripts/download_memgui3k.sh
bash scripts/restore_memgui3k_images.sh
bash scripts/build_memgui3k_training_data.sh
```

This creates:

```text
data/MemGUI-3K/training_data/
|-- train_sft.jsonl
`-- test_sft.jsonl
```

## Paper Configuration

The paper uses LoRA SFT on Qwen3-VL-8B-Instruct with the following key
settings:

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-8B-Instruct` |
| Train type | `lora` |
| Epochs | `1` |
| Learning rate | `1e-4` |
| LoRA rank | `8` |
| LoRA alpha | `32` |
| Target modules | `all-linear` |
| Max length | `32768` |
| Per-device train batch size | `2` |
| Per-device eval batch size | `1` |
| Gradient accumulation steps | `8` |
| Warmup ratio | `0.05` |
| Dataloader workers | `4` |
| GPUs | `8` |

## Train

The provided script is a public template. Adjust GPU count, cache directories,
model path, and dataset path for your environment.

```bash
bash training/ms_swift/train_memgui_8b_sft.sh
```

Useful environment variables:

| Variable | Default |
|---|---|
| `MODEL` | `Qwen/Qwen3-VL-8B-Instruct` |
| `TRAIN_FILE` | `data/MemGUI-3K/training_data/train_sft.jsonl` |
| `OUTPUT_DIR` | `outputs/memgui-sft-qwen3vl` |
| `MAX_LENGTH` | `32768` |
| `NPROC_PER_NODE` | `8` |
| `WANDB_PROJECT` | unset |
| `WANDB_API_KEY` | unset |

If you report to WandB, set `WANDB_API_KEY` in your shell or secret manager.
Do not commit real API keys.

Extra arguments are forwarded to `swift sft`.
