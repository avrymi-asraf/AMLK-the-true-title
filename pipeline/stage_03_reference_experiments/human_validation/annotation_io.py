"""Read-only helpers for the frozen human-validation worklist and annotations."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.common.paths import REFERENCE_EXPERIMENT_ARTIFACTS_DIR


HUMAN_VALIDATION_DIR = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "human_validation"
DEFAULT_WORKLIST_PATH = HUMAN_VALIDATION_DIR / "worklist.json"
TEAM_ANNOTATOR_IDS = ("amit", "avreymi", "ofek")


def default_annotations_path(annotator_id: str) -> Path:
    if annotator_id not in TEAM_ANNOTATOR_IDS:
        raise ValueError(f"unknown frozen annotator id: {annotator_id!r}")
    return HUMAN_VALIDATION_DIR / f"{annotator_id}.jsonl"


def default_team_annotation_paths() -> list[Path]:
    return [default_annotations_path(annotator_id) for annotator_id in TEAM_ANNOTATOR_IDS]


def load_worklist(path: str | Path = DEFAULT_WORKLIST_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        worklist = json.load(handle)
    if not isinstance(worklist, dict) or not isinstance(worklist.get("rows"), list):
        raise ValueError("human-validation worklist must contain a rows list")
    return worklist


def expand_tasks(worklist: dict, annotator_id: str | None = None) -> list[dict]:
    items = []
    for row in worklist.get("rows", []):
        if annotator_id and row.get("assigned_annotator") != annotator_id:
            continue
        for task in row.get("tasks", []):
            items.append({**row, "task": task})
    return items


def dedupe_annotations(records: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str, str], dict] = {}
    for record in records:
        key = (record["annotator_id"], record["hesum_id"], record["task"])
        previous = latest.get(key)
        if previous is None or record.get("submitted_at", "") >= previous.get("submitted_at", ""):
            latest[key] = record
    return list(latest.values())
