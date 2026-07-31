"""
Pipeline step 1 of 2: materialize the curated HeSum source for training.

The only training corpus is the main-branch data_curation product
(`final_clean_hesum.json`: rows of {hesum_id, text, headline}). This module
loads that JSON, normalizes each row to the pipeline contract
{text, summary, source, hesum_id}, and writes
`outputs/data/curated/curated_records.jsonl` for inspectability and side tools.

Curation itself (source filter + headline target rewrite) is NOT re-run here —
those artifacts live on the main worktree / were supplied offline. This repo
only consumes the clean result and turns it into a HuggingFace training dataset
via `data.preprocess`.

Run: python -m data.download [--input path/to/final_clean_hesum.json]
Execution environment: local development machine, CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_INPUT = Path("outputs/data/curated/final_clean_hesum.json")
OUTPUT_PATH = Path("outputs/data/curated/curated_records.jsonl")
SOURCE_LABEL = "hesum-curated"


def normalize_curated_record(record: dict) -> dict | None:
    """Map one curated row {hesum_id, text, headline} → train-facing fields.

    Returns None when text or headline is empty after strip (skip bad rows).
    """
    text = (record.get("text") or "").strip()
    headline = (record.get("headline") or "").strip()
    if not text or not headline:
        return None
    hesum_id = str(record.get("hesum_id") or "").strip()
    return {
        "text": text,
        "summary": headline,
        "source": SOURCE_LABEL,
        "hesum_id": hesum_id,
    }


def load_curated_json(path: Path) -> list[dict]:
    """Load final_clean_hesum.json (a JSON list) and normalize every usable row."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy final_clean_hesum.json from the main-branch "
            "data_curation artifacts (or place it at outputs/data/curated/)."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON list of records, got {type(raw).__name__}")

    records: list[dict] = []
    skipped = 0
    for row in raw:
        if not isinstance(row, dict):
            skipped += 1
            continue
        norm = normalize_curated_record(row)
        if norm is None:
            skipped += 1
            continue
        records.append(norm)
    if skipped:
        print(f"  Skipped {skipped} empty/invalid rows")
    return records


def write_records_jsonl(records: list[dict], path: Path) -> None:
    """Write normalized records as JSONL (one object per line, UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load curated HeSum JSON and write normalized curated_records.jsonl",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to final_clean_hesum.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Normalized JSONL path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        print(f"Output already exists at {args.output}. Pass --force to rebuild.")
        return 0

    print(f"Loading curated source from {args.input}...")
    try:
        records = load_curated_json(args.input)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"  Usable records: {len(records)}")
    write_records_jsonl(records, args.output)
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
