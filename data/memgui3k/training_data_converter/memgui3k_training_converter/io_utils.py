"""I/O helpers for MemGUI-3K training data conversion."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing text file: {path}")
    return path.read_text()


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def image_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing screenshot: {path}")
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{image_b64}"


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
