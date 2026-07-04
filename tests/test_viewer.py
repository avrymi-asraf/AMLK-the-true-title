"""Tests for evaluation.viewer.data: pure file-reading/filtering logic, no Streamlit/GPU/API.

Behavioral, not implementation-dictating — exercises the contracts evaluation/viewer/app.py
relies on: predictions load with <think> stripped, keyword search matches Hebrew substrings,
discovery finds only .jsonl files, and mismatched file lengths resolve to the shortest.
"""
import json

from evaluation.viewer import common_length, discover_predictions_files, filter_by_keyword, load_predictions


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_load_predictions_strips_think_block(tmp_path):
    path = tmp_path / "preds.jsonl"
    _write_jsonl(path, [
        {"text": "מאמר", "reference": "תקציר", "prediction": "<think>הרהור באנגלית</think>סיכום אמיתי",
         "model": "finetuned", "variant": "whole"},
    ])

    rows = load_predictions(path)

    assert rows[0]["prediction"] == "סיכום אמיתי"


def test_filter_by_keyword_matches_hebrew_substring_anywhere():
    rows = [
        {"text": "כתבה על ספורט", "prediction": "סיכום", "reference": "תקציר"},
        {"text": "כתבה על כלכלה", "prediction": "תחזית", "reference": "מחיר הדולר עלה"},
    ]

    assert filter_by_keyword(rows, "ספורט") == [0]
    assert filter_by_keyword(rows, "דולר") == [1]
    assert filter_by_keyword(rows, "לא קיים") == []
    assert filter_by_keyword(rows, "") == [0, 1]


def test_discover_predictions_files_only_finds_jsonl(tmp_path):
    (tmp_path / "predictions-finetuned.jsonl").write_text("")
    (tmp_path / "predictions-base.jsonl").write_text("")
    (tmp_path / "report.json").write_text("")
    (tmp_path / "notes.txt").write_text("")

    found = discover_predictions_files(str(tmp_path))

    assert [p.name for p in found] == ["predictions-base.jsonl", "predictions-finetuned.jsonl"]


def test_discover_predictions_files_missing_dir_returns_empty():
    assert discover_predictions_files("outputs/does-not-exist") == []


def test_common_length_returns_shortest_across_files():
    files_rows = {"finetuned": [{}] * 10, "base": [{}] * 7, "gemini": [{}] * 20}

    assert common_length(files_rows) == 7


def test_common_length_empty_input_is_zero():
    assert common_length({}) == 0
