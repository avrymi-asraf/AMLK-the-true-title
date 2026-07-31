"""Tests for data/download.py: curated HeSum → normalized train-facing records.

Behavioral checks only — does the curated loader produce the {text, summary,
source, hesum_id} contract preprocess and side tools depend on?
"""
import json
from pathlib import Path

import pytest

from data.download import load_curated_json, normalize_curated_record, write_records_jsonl


def test_normalize_curated_happy_path():
    row = {
        "hesum_id": "42",
        "text": "תוכן המאמר המלא בעברית",
        "headline": "כותרת מנוקה",
    }
    assert normalize_curated_record(row) == {
        "text": "תוכן המאמר המלא בעברית",
        "summary": "כותרת מנוקה",
        "source": "hesum-curated",
        "hesum_id": "42",
    }


def test_normalize_curated_strips_whitespace():
    row = {"hesum_id": "1", "text": "  טקסט  ", "headline": "  כותרת  "}
    result = normalize_curated_record(row)
    assert result["text"] == "טקסט"
    assert result["summary"] == "כותרת"


def test_normalize_curated_skips_empty_text():
    assert normalize_curated_record({"hesum_id": "1", "text": "", "headline": "כ"}) is None


def test_normalize_curated_skips_empty_headline():
    assert normalize_curated_record({"hesum_id": "1", "text": "ט", "headline": "  "}) is None


def test_load_curated_json_and_write_jsonl(tmp_path: Path):
    src = tmp_path / "final_clean_hesum.json"
    src.write_text(
        json.dumps([
            {"hesum_id": "1", "text": "מאמר א", "headline": "כותרת א"},
            {"hesum_id": "2", "text": "", "headline": "ריק"},  # skipped
            {"hesum_id": "3", "text": "מאמר ג", "headline": "כותרת ג"},
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    records = load_curated_json(src)
    assert len(records) == 2
    assert records[0]["hesum_id"] == "1"
    assert records[0]["source"] == "hesum-curated"

    out = tmp_path / "curated_records.jsonl"
    write_records_jsonl(records, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["summary"] == "כותרת א"


def test_load_curated_json_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_curated_json(tmp_path / "nope.json")
