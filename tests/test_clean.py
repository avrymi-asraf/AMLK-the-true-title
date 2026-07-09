"""Tests for data.clean: pure regex reference normalization, no model/API/GPU.

Behaviour covered: pipe/bullet digests become natural prose, leading list markers are dropped,
the result ends with a terminal period, cleaning is idempotent, and the roundup-digest filter
trips only on genuine multi-headline references.
"""
from data.clean import is_roundup_digest, normalize_summary, pipe_segments


def test_normalize_summary_rewrites_pipes_to_sentences():
    result = normalize_summary("כותרת אחת | כותרת שנייה | כותרת שלישית")
    assert "|" not in result
    assert result == "כותרת אחת. כותרת שנייה. כותרת שלישית."


def test_normalize_summary_rewrites_bullets():
    result = normalize_summary("• סעיף ראשון • סעיף שני")
    assert "•" not in result
    assert result == "סעיף ראשון. סעיף שני."


def test_normalize_summary_drops_leading_list_marker():
    assert normalize_summary("- פריט בודד").startswith("פריט")


def test_normalize_summary_ensures_terminal_period():
    assert normalize_summary("משפט ללא נקודה").endswith(".")
    # An existing terminal punctuation mark is preserved, not doubled.
    assert normalize_summary("שאלה פתוחה?").endswith("?")
    assert not normalize_summary("שאלה פתוחה?").endswith("?.")


def test_normalize_summary_is_idempotent_on_clean_prose():
    clean = "החתול ישב על המחצלת. הכלב ברח."
    assert normalize_summary(clean) == clean


def test_normalize_summary_handles_empty():
    assert normalize_summary("") == ""


def test_pipe_segments_counts_nonempty_segments():
    assert pipe_segments("א | ב | ג") == 3
    assert pipe_segments("משפט אחד ללא פייפ") == 1


def test_is_roundup_digest_trips_only_on_multi_headline():
    assert is_roundup_digest("א | ב | ג") is True
    assert is_roundup_digest("כותרת אחת | כותרת שנייה") is False  # 2 segments < default threshold 3
    assert is_roundup_digest("סיכום רגיל של כתבה אחת") is False
