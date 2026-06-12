"""Tests for data/preprocess.py: prompt building, probe variants, and splitting.

Behavioral checks only — does the data layer produce the contract the trainer and
the evaluation pipeline depend on (a prompt that names the task and carries the
article, lead/body variants that actually differ, a clean 80/10/10 split)?
"""
import datasets as hf_datasets
from data.preprocess import build_prompt, make_variant, split_dataset


def test_build_prompt_carries_task_and_text():
    result = build_prompt("המאמר הגדול")
    assert "Summarize" in result
    assert "Summary" in result
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
