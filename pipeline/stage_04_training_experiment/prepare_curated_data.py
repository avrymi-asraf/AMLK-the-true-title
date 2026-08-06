"""Normalize the curated HeSum artifact for the E4 curated training arm.

This module reads `artifacts/data_curation/final_clean_hesum.json`, whose rows contain
`{hesum_id, text, headline}`, normalizes them to the E4 training contract
`{text, summary, source, hesum_id}`, and writes
`outputs/training_experiment/curated/curated_records.jsonl`.

Curation itself is not rerun here. The normalized records become the curated-arm input to
`pipeline.stage_04_training_experiment.prepare_data`.

Run: python -m pipeline.stage_04_training_experiment.prepare_curated_data
Execution environment: local development machine, CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.common.paths import DATA_CURATION_ARTIFACTS_DIR, TRAINING_WORK_DIR

CANONICAL_INPUT = DATA_CURATION_ARTIFACTS_DIR / "final_clean_hesum.json"
OUTPUT_PATH = TRAINING_WORK_DIR / "curated" / "curated_records.jsonl"
SOURCE_LABEL = "hesum-curated"


def resolve_curated_input() -> Path:
    """Return the documented curated artifact path."""
    return CANONICAL_INPUT


DEFAULT_INPUT = resolve_curated_input()


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
            f"{path} not found. Place the frozen curated dataset at {CANONICAL_INPUT} "
            "or pass --input."
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
    records = load_curated_json(args.input)

    print(f"  Usable records: {len(records)}")
    write_records_jsonl(records, args.output)
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
