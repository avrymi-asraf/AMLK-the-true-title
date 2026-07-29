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
ANNOTATIONS_DIR = (
    Path(__file__).resolve().parents[2]
    / "data_curation"
    / "artifacts"
    / "human_annotations"
)

# Team annotator ids for the F9a blind round (sidebar preset in annotate_app.py).
TEAM_ANNOTATOR_IDS = ("amit", "avreymi", "ofek")


def default_annotations_path(annotator_id: str) -> Path:
    """Default JSONL path for one annotator's submissions (tracked in git)."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in annotator_id.strip()) or "annotator"
    return ANNOTATIONS_DIR / f"{safe_id}.jsonl"


def default_team_annotation_paths() -> list[Path]:
    """Paths for all team annotators — used by human_validation_results default."""
    return [default_annotations_path(aid) for aid in TEAM_ANNOTATOR_IDS]


def annotations_git_path(annotator_id: str) -> str:
    """Repo-relative path for git add/commit instructions."""
    return str(default_annotations_path(annotator_id).relative_to(
        Path(__file__).resolve().parents[2]
    ))


def load_worklist(path: str | Path = DEFAULT_WORKLIST_PATH) -> dict:
    """Load the frozen human-validation worklist."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_tasks(worklist: dict, annotator_id: str | None = None) -> list[dict]:
    """Flatten worklist rows into navigable task items: one entry per (hesum_id, task).

    When `annotator_id` is set, only rows assigned to that annotator are included (disjoint split).
    """
    items = []
    for row in worklist.get("rows", []):
        if annotator_id and row.get("assigned_annotator") != annotator_id:
            continue
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


def load_annotations(path: str | Path, *, dedupe: bool = True) -> list[dict]:
    """Read annotation records from JSONL (empty list if missing).

    When `dedupe` is True, keep only the latest record per (hesum_id, task) by submitted_at.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return dedupe_annotations(records) if dedupe else records


def dedupe_annotations(records: list[dict]) -> list[dict]:
    """Keep the latest record per (hesum_id, task)."""
    latest: dict[tuple[str, str], dict] = {}
    for rec in records:
        key = (rec["hesum_id"], rec["task"])
        prev = latest.get(key)
        if prev is None or rec.get("submitted_at", "") >= prev.get("submitted_at", ""):
            latest[key] = rec
    return list(latest.values())


def save_annotations(path: str | Path, records: list[dict]) -> None:
    """Rewrite a JSONL file from a list of records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def upsert_annotation(path: str | Path, record: dict) -> None:
    """Replace an existing (hesum_id, task) record or append if new."""
    path = Path(path)
    records = load_annotations(path, dedupe=False)
    key = (record["hesum_id"], record["task"])
    records = [r for r in records if (r["hesum_id"], r["task"]) != key]
    records.append(record)
    save_annotations(path, records)


def annotation_lookup(annotations: list[dict]) -> dict[tuple[str, str], dict]:
    """Map (hesum_id, task) -> record (expects deduped input)."""
    return {(a["hesum_id"], a["task"]): a for a in annotations}


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


def export_summary(
    annotations: list[dict],
    worklist: dict | None = None,
    annotator_id: str | None = None,
) -> dict:
    """Progress counts for the sidebar."""
    rubric_done = sum(1 for a in annotations if a["task"] == "rubric")
    pairwise_done = sum(1 for a in annotations if a["task"] == "pairwise")
    summary = {
        "rubric_done": rubric_done,
        "pairwise_done": pairwise_done,
        "total_done": len(annotations),
    }
    if worklist is not None:
        items = expand_tasks(worklist, annotator_id=annotator_id)
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
