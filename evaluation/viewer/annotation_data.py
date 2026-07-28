"""Human-validation annotation helpers for F9a — Streamlit-free load/save/resume logic.

Reads the frozen worklist (`human_validation_worklist.json`), tracks per-annotator JSONL
progress, and handles blind pairwise A/B slot assignment. Consumed by
`evaluation/viewer/annotate_app.py`; testable without Streamlit. Local, CPU-only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.rubric_judge import DIMENSIONS

DEFAULT_WORKLIST_PATH = (
    Path(__file__).resolve().parents[2]
    / "data_curation"
    / "artifacts"
    / "human_validation_worklist.json"
)

# Team annotator ids for the F9a blind round (sidebar preset in annotate_app.py).
TEAM_ANNOTATOR_IDS = ("amit", "avreymi", "ofek")


def default_annotations_path(annotator_id: str) -> Path:
    """Default JSONL path for one annotator's submissions."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in annotator_id.strip()) or "annotator"
    return Path(__file__).resolve().parents[2] / "outputs" / "results" / f"human_annotations_{safe_id}.jsonl"


def load_worklist(path: str | Path = DEFAULT_WORKLIST_PATH) -> dict:
    """Load the frozen human-validation worklist."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_tasks(worklist: dict) -> list[dict]:
    """Flatten worklist rows into navigable task items: one entry per (hesum_id, task)."""
    items = []
    for row in worklist.get("rows", []):
        for task in row.get("tasks", []):
            items.append({**row, "task": task})
    return items


def pairwise_presentation(
    annotator_id: str,
    hesum_id: str,
    original: str,
    curated: str,
) -> dict:
    """Deterministic blind A/B assignment per annotator (stable on resume).

    Returns headline_a, headline_b, slot_map {a: original|curated, b: ...}, curated_is_a.
    """
    key = f"{annotator_id}:{hesum_id}".encode()
    digest = hashlib.sha256(key).hexdigest()
    swap = int(digest[:8], 16) % 2 == 1
    if swap:
        return {
            "headline_a": curated,
            "headline_b": original,
            "slot_map": {"a": "curated", "b": "original"},
            "curated_is_a": True,
        }
    return {
        "headline_a": original,
        "headline_b": curated,
        "slot_map": {"a": "original", "b": "curated"},
        "curated_is_a": False,
    }


def load_annotations(path: str | Path) -> list[dict]:
    """Read all annotation records from a JSONL file (empty list if missing)."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def completed_keys(annotations: list[dict]) -> set[tuple[str, str]]:
    """Set of (hesum_id, task) pairs already submitted."""
    return {(a["hesum_id"], a["task"]) for a in annotations}


def append_annotation(path: str | Path, record: dict) -> None:
    """Append one annotation record as a single JSONL line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_rubric_record(
    annotator_id: str,
    hesum_id: str,
    scores: dict[str, int],
) -> dict:
    """Validate and package a rubric submission."""
    for dim in DIMENSIONS:
        if dim not in scores or scores[dim] not in range(1, 6):
            raise ValueError(f"Invalid or missing score for {dim}")
    return {
        "hesum_id": hesum_id,
        "annotator_id": annotator_id,
        "task": "rubric",
        "scores": {dim: int(scores[dim]) for dim in DIMENSIONS},
        "submitted_at": utc_now_iso(),
    }


def build_pairwise_record(
    annotator_id: str,
    hesum_id: str,
    winner: str,
    slot_map: dict[str, str],
) -> dict:
    """Validate and package a pairwise submission (winner is a|b|tie)."""
    winner = winner.lower().strip()
    if winner not in {"a", "b", "tie"}:
        raise ValueError(f"Invalid winner: {winner}")
    return {
        "hesum_id": hesum_id,
        "annotator_id": annotator_id,
        "task": "pairwise",
        "winner": winner,
        "slot_map": slot_map,
        "submitted_at": utc_now_iso(),
    }


def export_summary(annotations: list[dict], worklist: dict | None = None) -> dict:
    """Progress counts for the sidebar."""
    rubric_done = sum(1 for a in annotations if a["task"] == "rubric")
    pairwise_done = sum(1 for a in annotations if a["task"] == "pairwise")
    summary = {
        "rubric_done": rubric_done,
        "pairwise_done": pairwise_done,
        "total_done": len(annotations),
    }
    if worklist is not None:
        items = expand_tasks(worklist)
        summary["rubric_total"] = sum(1 for i in items if i["task"] == "rubric")
        summary["pairwise_total"] = sum(1 for i in items if i["task"] == "pairwise")
        summary["total_tasks"] = len(items)
    return summary


def filter_task_items(
    items: list[dict],
    completed: set[tuple[str, str]],
    *,
    task_filter: str = "all",
    only_remaining: bool = False,
) -> list[dict]:
    """Filter navigable items by task type and/or completion status."""
    filtered = items
    if task_filter in {"rubric", "pairwise"}:
        filtered = [i for i in filtered if i["task"] == task_filter]
    if only_remaining:
        filtered = [
            i for i in filtered
            if (i["hesum_id"], i["task"]) not in completed
        ]
    return filtered
