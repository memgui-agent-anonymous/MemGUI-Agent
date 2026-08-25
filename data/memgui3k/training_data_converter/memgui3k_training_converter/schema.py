"""Schema helpers for MemGUI-3K training samples."""

from __future__ import annotations

from pathlib import Path


def validate_dataset_layout(dataset_dir: Path, splits: list[str]) -> None:
    required = [dataset_dir / "system_prompt.txt", dataset_dir / "images"]
    required.extend(dataset_dir / f"{split}_trajectories.jsonl" for split in splits)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset path: {path}")


def make_training_sample(
    system_prompt: str,
    user_prompt: str,
    assistant_response: str,
    image_url: str | None,
    task_id: str,
    step,
) -> dict:
    user_content = f"<image>{user_prompt}" if image_url else user_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_response},
    ]

    sample = {"messages": messages}
    if image_url:
        sample["images"] = [image_url]
    sample["task_id"] = task_id
    sample["step"] = step
    return sample
