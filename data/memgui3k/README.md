# MemGUI-3K Data Utilities

This directory contains public utilities for working with the MemGUI-3K
Hugging Face dataset.

The dataset repository itself only stores dataset metadata, trajectory JSONL
files, task index metadata, and split screenshot archives. Data restoration and
conversion scripts live in this MemGUI-Agent repository.

## Download

From the repository root:

```bash
bash scripts/download_memgui3k.sh
```

By default this downloads `memgui-agent-anonymous/MemGUI-3K` into
`data/MemGUI-3K`.

Custom location:

```bash
MEMGUI3K_DIR=/path/to/MemGUI-3K bash scripts/download_memgui3k.sh
```

## Restore Screenshots

The Hugging Face release stores screenshots in:

```text
image_archives/
|-- images.z01
|-- images.z02
|-- ...
`-- images.zip
```

Restore them into `images/`:

```bash
bash scripts/restore_memgui3k_images.sh
```

The script first converts the split archive to a regular zip with `zip -s 0`,
then unzips it into the dataset root. After restoration, trajectory screenshot
paths such as `images/001-M01_attempt_2_step_1.png` resolve directly.

## Build Training JSONL

MemGUI-3K does not store `train_sft.jsonl` or `test_sft.jsonl` on Hugging Face.
Build them from the released trajectories and restored screenshots:

```bash
bash scripts/build_memgui3k_training_data.sh
```

Output:

```text
data/MemGUI-3K/training_data/
|-- train_sft.jsonl
`-- test_sft.jsonl
```

Quick smoke test:

```bash
python3 data/memgui3k/training_data_converter/build_training_data.py \
  --dataset-dir data/MemGUI-3K \
  --output-dir /tmp/memgui3k_training_smoke \
  --max-trajectories 2
```

## Prepare A Hugging Face Upload Directory

If you have a complete local MemGUI-3K directory with a real `images/` folder,
you can prepare an upload-friendly dataset directory:

```bash
bash scripts/package_memgui3k_hf_release.sh \
  /path/to/MemGUI-3K \
  /path/to/MemGUI-3K-HF-Release
```

The output directory excludes the raw `images/` folder and stores screenshots
as 10GB split zip files under `image_archives/`.
