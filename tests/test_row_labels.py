"""Tests for data_curation.analysis.row_labels: pure join/classification logic, no tokenizer,
no downloads, no API. The tokenizer-dependent counting (`build_token_counts`) is exercised by
actually running the pipeline, not here — see AGENTS.md status notes.
"""
import pytest

from data_curation.analysis.row_labels import (
    build_row_labels,
    classify_headline_edit,
    compute_lead_overlap,
    count_pipes,
    validate_counts,
)


def test_count_pipes():
    assert count_pipes("כותרת אחת | כותרת שנייה") == 1
    assert count_pipes("כותרת רגילה") == 0


def test_compute_lead_overlap_full_and_none():
    article = "הכלב רץ בפארק אתמול בבוקר"
    assert compute_lead_overlap(article, "הכלב רץ בפארק") == 1.0
    assert compute_lead_overlap(article, "החתול ישן בבית") == 0.0
    assert compute_lead_overlap(article, "") == 0.0


def test_classify_headline_edit_pipes_removed():
    assert classify_headline_edit("ידיעה אחת | ידיעה שנייה", "ידיעה אחת") == "pipes_removed"


def test_classify_headline_edit_full_rewrite_on_low_overlap():
    assert classify_headline_edit("החתול ישן על הספה", "ראש הממשלה נאם בכנסת") == "full_rewrite"


def test_classify_headline_edit_light_edit_on_high_overlap():
    # Same word count and a single word swapped: high overlap, but not a prefix/substring trim.
    assert classify_headline_edit(
        "ראש הממשלה נאם אתמול בכנסת",
        "ראש הממשלה דיבר אתמול בכנסת",
    ) == "light_edit"


def test_build_row_labels_joins_every_artifact_by_id():
    records = [
        {"id": "1", "text": "טקסט אחד", "headline": "כותרת אחת", "tail_boilerplate_removed": False},
        {"id": "2", "text": "טקסט שני", "headline": "כותרת ראשונה | כותרת שנייה", "tail_boilerplate_removed": True},
        {"id": "3", "text": "טקסט שלישי", "headline": "כותרת שלישית", "tail_boilerplate_removed": False},
    ]
    token_counts = {"1": (100, 5), "2": (200, 8), "3": (50, 4)}
    token_budget_filter = {"1": True, "2": True, "3": False}  # id 3 over budget
    headline_pipe_filter = {"1": True, "2": False, "3": True}  # id 2 multi-pipe
    source_labels = {"1": "usable", "2": "unusable_other"}  # id 3 never reached model curation
    headline_replacements = {"1": None}  # id 1 kept as-is; id 2/3 never reached headline curation

    rows = build_row_labels(
        records, token_counts, token_budget_filter, headline_pipe_filter,
        source_labels, headline_replacements,
    )
    by_id = {row["hesum_id"]: row for row in rows}

    assert by_id["1"]["over_token_budget"] is False
    assert by_id["1"]["multi_pipe"] is False
    assert by_id["1"]["reached_model_curation"] is True
    assert by_id["1"]["headline_action"] == "kept"
    assert by_id["1"]["headline_edit_type"] is None

    assert by_id["2"]["multi_pipe"] is True
    assert by_id["2"]["n_pipes"] == 1
    assert by_id["2"]["source_label"] == "unusable_other"
    assert by_id["2"]["headline_action"] is None  # unusable rows never reach headline curation

    assert by_id["3"]["over_token_budget"] is True
    assert by_id["3"]["reached_model_curation"] is False
    assert by_id["3"]["source_label"] is None


def test_validate_counts_raises_on_mismatch():
    with pytest.raises(ValueError):
        validate_counts([{
            "tail_boilerplate_removed": False, "over_token_budget": False, "multi_pipe": False,
            "reached_model_curation": True, "source_label": "usable", "headline_action": "kept",
        }])
