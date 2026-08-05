"""JSON loading and saving helpers used by the curation pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    """Write UTF-8 JSON atomically and tolerate identical locked outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    try:
        os.replace(temp_path, path)
    except PermissionError as error:
        if path.exists() and temp_path.read_bytes() == path.read_bytes():
            temp_path.unlink()
            return

        raise PermissionError(
            f"Could not replace {path}. Close any program that has this JSON file open "
            f"and rerun the command. The temporary file was kept at {temp_path}."
        ) from error
