"""Tests for data/download.py normalization functions."""
from data.download import normalize_iahlt, normalize_hesum


def test_normalize_iahlt_happy_path():
    record = {
        "text_raw": "כותרת המאמר וכל התוכן שלו בעברית",
        "summary": "סיכום קצר",
        "metadata": {"source": "haaretz", "doc_id": "123"},
    }
    result = normalize_iahlt(record)
    assert result == {
        "text": "כותרת המאמר וכל התוכן שלו בעברית",
        "summary": "סיכום קצר",
        "source": "iahlt",
    }


def test_normalize_iahlt_skips_empty_text():
    assert normalize_iahlt({"text_raw": "", "summary": "סיכום", "metadata": {}}) is None


def test_normalize_iahlt_skips_empty_summary():
    assert normalize_iahlt({"text_raw": "טקסט", "summary": "", "metadata": {}}) is None


def test_normalize_hesum_happy_path():
    record = {"article": "תוכן המאמר המלא", "summary": "הכותרת"}
    result = normalize_hesum(record)
    assert result == {"text": "תוכן המאמר המלא", "summary": "הכותרת", "source": "hesum"}


def test_normalize_hesum_skips_empty_article():
    assert normalize_hesum({"article": "", "summary": "סיכום"}) is None
