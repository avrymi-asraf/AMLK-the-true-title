"""Behavior tests for the local E4 paper-figure input path.

The figure stage follows E4 scoring and turns paired raw/curated rubric rows into the paper's
four-dimension ordinal distributions. These tests keep that CPU-only reporting boundary honest
without loading a model, calling an API, or pinning Plotly layout details.
"""

from __future__ import annotations

import json

import pytest

from data_curation.analysis.e4_figures import load_rubric_rows, summarize_rubric_rows


def _row(raw: list[int], curated: list[int]) -> dict:
    dimensions = ("faithfulness", "single_focus", "informativeness", "cleanliness")
    return {
        "text_prefix": "article",
        "raw_scores": dict(zip(dimensions, raw)),
        "curated_scores": dict(zip(dimensions, curated)),
    }


def test_summarize_rubric_rows_keeps_all_dimensions_and_both_arms(tmp_path):
    path = tmp_path / "scores.jsonl"
    rows = [
        _row([1, 2, 3, 4], [2, 2, 4, 5]),
        _row([5, 4, 3, 2], [5, 5, 3, 1]),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_rubric_rows(load_rubric_rows(path))

    assert summary["n_pairs"] == 2
    assert set(summary["by_dimension"]) == {
        "faithfulness",
        "single_focus",
        "informativeness",
        "cleanliness",
    }
    assert summary["by_dimension"]["faithfulness"]["raw_counts"] == [1, 0, 0, 0, 1]
    assert summary["by_dimension"]["faithfulness"]["curated_counts"] == [0, 1, 0, 0, 1]
    assert summary["by_dimension"]["cleanliness"]["raw_mean"] == 3.0
    assert summary["by_dimension"]["cleanliness"]["curated_mean"] == 3.0


def test_load_rubric_rows_rejects_missing_or_out_of_range_scores(tmp_path):
    path = tmp_path / "bad-scores.jsonl"
    path.write_text(
        json.dumps(_row([1, 2, 3, 4], [2, 2, 4, 6])) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scores must be integers from 1 to 5"):
        load_rubric_rows(path)
