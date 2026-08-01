"""Tests for data/preprocess.py: prompt building, probe variants, splitting, E4 test swap.

Behavioral checks only — does the data layer produce the contract the trainer and
the evaluation pipeline depend on (a prompt that names the task and carries the
article, lead/body variants that actually differ, a clean 80/10/10 split, and
post-swap validation for E4)?
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import datasets as hf_datasets
import pytest

from data.prompts import build_prompt
from data.preprocess import (
    build_train_dataset,
    load_test_split,
    make_variant,
    split_dataset,
    validate_train_dataset,
)


class _FakeTok:
    """Minimal tokenizer stub: one 'token' per character for truncate tests."""

    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = list(range(len(text)))
        return R()

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)


def test_build_prompt_carries_task_and_text():
    result = build_prompt("המאמר הגדול")
    assert "סכם" in result
    assert "תקציר" in result
    assert "המאמר הגדול" in result
    assert "35" in result  # E4 35-word budget


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
        "source": ["iahlt"] * 500 + ["hesum"] * 500,
    })
    train, val, test = split_dataset(data, seed=42)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_split_dataset_no_overlap():
    data = hf_datasets.Dataset.from_dict({
        "text": [f"text {i}" for i in range(100)],
        "summary": [f"summary {i}" for i in range(100)],
        "source": ["iahlt"] * 100,
    })
    train, val, test = split_dataset(data, seed=42)
    train_set, val_set, test_set = set(train["text"]), set(val["text"]), set(test["text"])
    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set


def _make_split_rows(prefix: str, n: int) -> hf_datasets.Dataset:
    texts = [f"{prefix} article body {i}" for i in range(n)]
    summaries = [f"{prefix} head {i}" for i in range(n)]
    return hf_datasets.Dataset.from_dict({
        "text": texts,
        "summary": summaries,
        "source": ["hesum-curated"] * n,
        "prompt": [build_prompt(t) for t in texts],
        "completion": list(summaries),
    })


def test_validate_train_dataset_accepts_clean_splits():
    train = _make_split_rows("train", 8)
    val = _make_split_rows("val", 2)
    test = _make_split_rows("test", 2)
    validate_train_dataset(train, val, test)  # no raise


def test_validate_train_dataset_rejects_train_test_text_overlap():
    train = _make_split_rows("shared", 4)
    val = _make_split_rows("val", 2)
    test = _make_split_rows("shared", 2)  # same text prefix → same strings as train[:2]
    # Force exact overlap on one text.
    test = test.map(lambda row, idx: {"text": train["text"][0]} if idx == 0 else row,
                    with_indices=True)
    # Rebuild prompt so contract still has text-in-prompt (only overlap check should fire).
    test = test.map(lambda row: {"prompt": build_prompt(row["text"]),
                                 "completion": row["summary"]})
    with pytest.raises(ValueError, match="overlap"):
        validate_train_dataset(train, val, test)


def test_test_from_swap_revalidates_and_shares_test(tmp_path: Path):
    """E4: after swap, test equals the curated test and post-swap validate runs."""
    curated_test = _make_split_rows("curated-test", 3)
    curated_dir = tmp_path / "e4cur"
    curated_dir.mkdir()
    curated_test.save_to_disk(str(curated_dir / "test"))

    loaded = load_test_split(curated_dir)
    assert list(loaded["text"]) == list(curated_test["text"])

    # Raw arm train/val (disjoint texts) + swapped curated test.
    train = _make_split_rows("raw-train", 8)
    val = _make_split_rows("raw-val", 2)
    validate_train_dataset(train, val, loaded)

    # Byte-identical: reloading again matches.
    loaded2 = load_test_split(curated_dir)
    assert list(loaded["text"]) == list(loaded2["text"])
    assert list(loaded["summary"]) == list(loaded2["summary"])


def test_test_from_swap_catches_overlap_on_revalidate(tmp_path: Path):
    """Post-swap validate must catch raw-train text that equals curated-test text."""
    curated_test = _make_split_rows("leak", 2)
    curated_dir = tmp_path / "e4cur"
    curated_dir.mkdir()
    curated_test.save_to_disk(str(curated_dir / "test"))
    swapped = load_test_split(curated_dir)

    train = _make_split_rows("leak", 4)  # same texts as curated test
    val = _make_split_rows("raw-val-ok", 2)
    with pytest.raises(ValueError, match="overlap"):
        validate_train_dataset(train, val, swapped)


def test_build_train_dataset_source_preserved():
    records = [
        {"text": f"body {i}", "summary": f"head {i}", "source": "hesum-raw", "hesum_id": str(i)}
        for i in range(5)
    ]
    ds = build_train_dataset(records, "whole", _FakeTok())
    assert set(ds["source"]) == {"hesum-raw"}
    assert "prompt" in ds.column_names
    assert ds["completion"] == ds["summary"]


def test_main_calls_validate_after_test_from(tmp_path: Path):
    """CLI path: --test-from triggers a second validate_train_dataset call."""
    from data import preprocess as pp

    # Minimal records so build + split work without the real tokenizer.
    records = [
        {
            "text": f"article body number {i} with enough words",
            "summary": f"headline {i}",
            "source": "hesum-raw",
            "hesum_id": str(i),
        }
        for i in range(20)
    ]
    curated_test = _make_split_rows("curated-test", 2)
    curated_dir = tmp_path / "cur"
    curated_dir.mkdir()
    curated_test.save_to_disk(str(curated_dir / "test"))
    out_dir = tmp_path / "e4raw"

    calls: list[tuple] = []
    real_validate = pp.validate_train_dataset

    def tracking_validate(train, val, test):
        calls.append((len(train), len(val), len(test), list(test["text"])[:1]))
        return real_validate(train, val, test)

    class _Tok(_FakeTok):
        pass

    with patch.object(pp, "load_source_records", return_value=records), \
         patch.object(pp, "validate_train_dataset", side_effect=tracking_validate), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tok()):
        rc = pp.main([
            "--variant", "whole",
            "--force",
            "--output", str(out_dir),
            "--test-from", str(curated_dir),
            "--seed", "42",
        ])
    assert rc == 0
    assert len(calls) == 2, "expected pre-swap + post-swap validate"
    # Second call must use the curated test texts.
    assert calls[1][3] == [curated_test["text"][0]]
    assert (out_dir / "test").exists()
