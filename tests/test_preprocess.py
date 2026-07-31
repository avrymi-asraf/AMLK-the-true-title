"""Tests for data/preprocess.py: prompt building, probe variants, split, train contract.

Behavioral checks only — does the data layer produce the contract the trainer and
the evaluation pipeline depend on (prompt carries the article, lead/body differ,
80/10/10 split, validate_train_dataset accepts a good build and rejects a bad one)?
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import datasets as hf_datasets
import pytest

from data.preprocess import (
    TRAIN_COLUMNS,
    build_train_dataset,
    load_source_records,
    split_dataset,
    validate_train_dataset,
)
from data.prompts import build_prompt, make_variant


def test_build_prompt_carries_task_and_text():
    result = build_prompt("המאמר הגדול")
    assert "סכם" in result
    assert "תקציר" in result
    assert "המאמר הגדול" in result


def test_make_variant_whole_is_identity():
    text = "פסקה ראשונה\n\nפסקה שנייה\n\nפסקה שלישית"
    assert make_variant(text, "whole") == text


def test_make_variant_lead_and_body_partition_paragraphs():
    text = "פסקה ראשונה\n\nפסקה שנייה\n\nפסקה שלישית"
    lead = make_variant(text, "lead")
    body = make_variant(text, "body")
    assert lead == "פסקה ראשונה"
    assert "פסקה ראשונה" not in body
    assert "פסקה שנייה" in body


def test_make_variant_falls_back_to_sentences_without_paragraphs():
    text = "המשפט הראשון. המשפט השני. המשפט השלישי."
    lead = make_variant(text, "lead")
    body = make_variant(text, "body")
    assert lead == "המשפט הראשון."
    assert "המשפט השני" in body
    assert "המשפט הראשון" not in body


def test_split_dataset_ratios():
    data = hf_datasets.Dataset.from_dict({
        "text": [f"text {i}" for i in range(1000)],
        "summary": [f"summary {i}" for i in range(1000)],
        "source": ["hesum-curated"] * 1000,
    })
    train, val, test = split_dataset(data, seed=42)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_split_dataset_no_overlap():
    data = hf_datasets.Dataset.from_dict({
        "text": [f"text {i}" for i in range(100)],
        "summary": [f"summary {i}" for i in range(100)],
        "source": ["hesum-curated"] * 100,
    })
    train, val, test = split_dataset(data, seed=42)
    train_set, val_set, test_set = set(train["text"]), set(val["text"]), set(test["text"])
    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set


def _identity_tokenizer():
    """Tokenizer stub: encode = char-ish ids, decode = join (no real model)."""
    tok = MagicMock()

    def _call(text, add_special_tokens=False):
        # One "token" per character is fine for truncation tests.
        m = MagicMock()
        m.input_ids = list(range(len(text)))
        return m

    tok.side_effect = _call
    tok.decode = lambda ids, skip_special_tokens=True: "x" * len(ids)
    return tok


def test_build_train_dataset_columns_and_completion_match_summary():
    records = [
        {"text": "מאמר ארוך על פוליטיקה", "summary": "כותרת א", "source": "hesum-curated", "hesum_id": "1"},
        {"text": "מאמר על ספורט היום", "summary": "כותרת ב", "source": "hesum-curated", "hesum_id": "2"},
    ]
    ds = build_train_dataset(records, "whole", _identity_tokenizer())
    assert list(ds.column_names) == list(TRAIN_COLUMNS)
    assert ds["completion"] == ds["summary"]
    assert "מאמר ארוך על פוליטיקה" in ds["prompt"][0]
    assert ds["summary"][0] == "כותרת א"


def test_validate_train_dataset_accepts_good_splits():
    records = [
        {
            "text": f"מאמר מספר {i} עם תוכן מספיק ארוך",
            "summary": f"כותרת {i}",
            "source": "hesum-curated",
            "hesum_id": str(i),
        }
        for i in range(20)
    ]
    ds = build_train_dataset(records, "whole", _identity_tokenizer())
    train, val, test = split_dataset(ds, seed=0)
    validate_train_dataset(train, val, test)  # must not raise


def test_validate_train_dataset_rejects_completion_mismatch():
    train = hf_datasets.Dataset.from_dict({
        "text": ["a"],
        "summary": ["s"],
        "source": ["hesum-curated"],
        "prompt": [build_prompt("a")],
        "completion": ["DIFFERENT"],
    })
    val = train
    test = train
    with pytest.raises(ValueError, match="completion must equal summary"):
        validate_train_dataset(train, val, test)


def test_validate_train_dataset_rejects_missing_column():
    train = hf_datasets.Dataset.from_dict({
        "text": ["a"],
        "summary": ["s"],
        "source": ["hesum-curated"],
        "prompt": [build_prompt("a")],
        # missing completion
    })
    with pytest.raises(ValueError, match="missing required columns"):
        validate_train_dataset(train, train, train)


def test_load_source_records_from_curated_json(tmp_path: Path):
    path = tmp_path / "final_clean_hesum.json"
    path.write_text(
        json.dumps([
            {"hesum_id": "9", "text": "גוף הכתבה", "headline": "כותרת מנוקה"},
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    rows = load_source_records(path)
    assert len(rows) == 1
    assert rows[0]["summary"] == "כותרת מנוקה"
    assert rows[0]["source"] == "hesum-curated"
