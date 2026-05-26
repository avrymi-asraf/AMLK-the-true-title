"""Tests for data/preprocess.py formatting and splitting functions."""
import datasets as hf_datasets
from data.preprocess import format_instruction, split_dataset


def test_format_instruction_contains_text_and_summary():
    result = format_instruction("המאמר הגדול", "סיכום קצר")
    assert "המאמר הגדול" in result
    assert "סיכום קצר" in result


def test_format_instruction_contains_prompt_and_label():
    result = format_instruction("א", "ב")
    assert "Summarize" in result
    assert "Summary" in result


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
    train_set = set(train["text"])
    val_set = set(val["text"])
    test_set = set(test["text"])
    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set
