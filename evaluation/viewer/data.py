"""
Evaluation pipeline, results-reading step: plain, Streamlit-free helpers for browsing a
predictions.jsonl file (the {text, reference, prediction, model, variant} rows produced by
predict.py / train_hf_job.py / infer.py). Kept import-clean of Streamlit so these functions stay
usable from a notebook or REPL, not just the UI. `evaluation/viewer/app.py` is the only consumer
of these functions for the interactive viewer; it adds no data logic of its own. Re-exported by
`evaluation/viewer/__init__.py` so callers can just `from evaluation.viewer import ...`.

Execution environment: local machine, CPU-only, no GPU/API — this only reads files already on
disk under outputs/results/.
"""
import json
from pathlib import Path

from evaluation.gemini_client import strip_think


def discover_predictions_files(results_dir: str = "outputs/results") -> list[Path]:
    """List *.jsonl files directly under results_dir, sorted by name.

    Non-recursive, so the .cache/ subdir (HF datasets cache) is never picked up.
    """
    root = Path(results_dir)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.jsonl"))


def load_predictions(path: str | Path) -> list[dict]:
    """Read a predictions.jsonl file, stripping <think> blocks from each prediction.

    Mirrors evaluate.py's preprocessing so the viewer shows exactly what gets scored.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["prediction"] = strip_think(row["prediction"])
            rows.append(row)
    return rows


def filter_by_keyword(rows: list[dict], keyword: str) -> list[int]:
    """Indices of rows where keyword appears (case-insensitive) in text/prediction/reference.

    An empty/whitespace-only keyword matches every row.
    """
    keyword = keyword.strip().lower()
    if not keyword:
        return list(range(len(rows)))
    return [
        i for i, row in enumerate(rows)
        if keyword in row.get("text", "").lower()
        or keyword in row.get("prediction", "").lower()
        or keyword in row.get("reference", "").lower()
    ]


def common_length(files_rows: dict[str, list[dict]]) -> int:
    """Shortest row count across the given {label: rows} files, 0 if none given."""
    if not files_rows:
        return 0
    return min(len(rows) for rows in files_rows.values())
