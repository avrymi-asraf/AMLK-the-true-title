"""
Pipeline step 1 of 3: dataset acquisition.
Downloads Hebrew summarization datasets from HuggingFace Hub (IAHLT/summarization_he
and biunlp/HeSum), normalises their schemas to {text, summary, source}, and writes
the merged dataset to outputs/data/raw/combined.jsonl.

Run: python -m data.download
Execution environment: local development machine with HF_TOKEN in environment.
"""
import json
import os
import sys
from pathlib import Path

import datasets


OUTPUT_PATH = Path("outputs/data/raw/combined.jsonl")


def normalize_iahlt(record: dict) -> dict | None:
    """Return {text, summary, source='iahlt'} from an IAHLT JSONL record, or None to skip."""
    text = record.get("text_raw", "").strip()
    summary = record.get("summary", "").strip()
    if not text or not summary:
        return None
    return {"text": text, "summary": summary, "source": "iahlt"}


def normalize_hesum(record: dict) -> dict | None:
    """Return {text, summary, source='hesum'} from a HeSum HF row, or None to skip."""
    text = record.get("article", "").strip()
    summary = record.get("summary", "").strip()
    if not text or not summary:
        return None
    return {"text": text, "summary": summary, "source": "hesum"}


def _load_iahlt(hf_token: str) -> list[dict]:
    print("Loading IAHLT/summarization_he from HuggingFace Hub...")
    try:
        ds = datasets.load_dataset("IAHLT/summarization_he", token=hf_token)
    except Exception as e:
        print(
            f"  WARNING: Could not load IAHLT/summarization_he ({e}). "
            "Skipping — dataset may require special access or gating approval.",
            file=sys.stderr,
        )
        return []
    records = []
    for split in ds.values():
        for row in split:
            norm = normalize_iahlt(row)
            if norm:
                records.append(norm)
    print(f"  IAHLT: {len(records)} usable records")
    return records


def _load_hesum() -> list[dict]:
    print("Loading biunlp/HeSum from HuggingFace Hub...")
    ds = datasets.load_dataset("biunlp/HeSum")
    records = []
    for split in ds.values():
        for row in split:
            norm = normalize_hesum(row)
            if norm:
                records.append(norm)
    print(f"  HeSum: {len(records)} usable records")
    return records


def main():
    if OUTPUT_PATH.exists():
        print(f"Output already exists at {OUTPUT_PATH}. Delete it to re-download.")
        sys.exit(0)

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = _load_iahlt(hf_token) + _load_hesum()
    print(f"Total: {len(records)} records")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
