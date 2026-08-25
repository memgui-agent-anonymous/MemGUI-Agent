"""Conversion from released MemGUI-3K trajectories to training samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .io_utils import image_to_data_url, iter_jsonl, load_text, write_jsonl
from .schema import make_training_sample, validate_dataset_layout


def iter_training_samples(
    dataset_dir: Path,
    split: str,
    system_prompt: str,
    max_trajectories: int | None = None,
) -> Iterable[dict]:
    """Yield step-level training samples for one split."""

    trajectories_path = dataset_dir / f"{split}_trajectories.jsonl"
    for traj_idx, trajectory in enumerate(iter_jsonl(trajectories_path), 1):
        if max_trajectories is not None and traj_idx > max_trajectories:
            break

        task_id = trajectory.get("task_id", "")
        for step in trajectory.get("steps", []):
            if not step.get("is_reasonable"):
                continue

            assistant_response = step.get("assistant_response") or ""
            if len(assistant_response.strip()) < 10:
                continue

            screenshot = step.get("screenshot")
            image_url = None
            if screenshot:
                image_url = image_to_data_url(dataset_dir / screenshot)

            yield make_training_sample(
                system_prompt=system_prompt,
                user_prompt=step.get("user_prompt") or "",
                assistant_response=assistant_response,
                image_url=image_url,
                task_id=task_id,
                step=step.get("step"),
            )


def convert_split(
    dataset_dir: Path,
    output_dir: Path,
    split: str,
    system_prompt: str,
    max_trajectories: int | None,
    overwrite: bool,
) -> dict:
    """Convert one split and return summary statistics."""

    output_path = output_dir / f"{split}_sft.jsonl"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    samples = iter_training_samples(
        dataset_dir=dataset_dir,
        split=split,
        system_prompt=system_prompt,
        max_trajectories=max_trajectories,
    )
    n_samples = write_jsonl(output_path, samples)

    n_trajectories = 0
    for _ in iter_jsonl(dataset_dir / f"{split}_trajectories.jsonl"):
        n_trajectories += 1
        if max_trajectories is not None and n_trajectories >= max_trajectories:
            break

    return {
        "trajectories": n_trajectories,
        "samples": n_samples,
        "output_path": str(output_path),
    }


def convert_dataset(
    dataset_dir: Path,
    output_dir: Path,
    splits: list[str],
    max_trajectories: int | None = None,
    overwrite: bool = True,
) -> dict[str, dict]:
    """Convert selected splits from a released MemGUI-3K folder."""

    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    validate_dataset_layout(dataset_dir, splits)
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_text(dataset_dir / "system_prompt.txt")
    stats = {}
    for split in splits:
        stats[split] = convert_split(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            split=split,
            system_prompt=system_prompt,
            max_trajectories=max_trajectories,
            overwrite=overwrite,
        )
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )
    return stats
