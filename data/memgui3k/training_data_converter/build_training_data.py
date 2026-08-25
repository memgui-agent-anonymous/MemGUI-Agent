#!/usr/bin/env python3
"""Build MemGUI-3K step-level training JSONL from released trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from memgui3k_training_converter.convert import convert_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="MemGUI-3K root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <dataset-dir>/training_data.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Splits to convert.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Process at most N trajectories per split for smoke tests.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if output JSONL files already exist.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir or (dataset_dir / "training_data")
    stats = convert_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        splits=args.splits,
        max_trajectories=args.max_trajectories,
        overwrite=not args.no_overwrite,
    )

    print("MemGUI-3K training data conversion complete.")
    for split, split_stats in stats.items():
        print(
            f"{split}: {split_stats['trajectories']} trajectories, "
            f"{split_stats['samples']} samples -> {split_stats['output_path']}"
        )


if __name__ == "__main__":
    main()
