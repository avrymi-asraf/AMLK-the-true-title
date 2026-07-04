"""Tests for evaluation.style_labels: pure regex classification, no model/API calls, no GPU."""
from evaluation.style_labels import MULTI_SENTENCE, PIPE_DIGEST, QUESTION, SINGLE_SENTENCE, classify_style, style_summary


def test_classify_style_detects_pipe_digest():
    assert classify_style('כותרת אחת | כותרת שנייה | כותרת שלישית') == PIPE_DIGEST


def test_classify_style_detects_question_over_sentence_count():
    # A question with an internal period must not be misread as a plain multi-sentence summary.
    assert classify_style('מה קרה היום. ולמה זה חשוב?') == QUESTION


def test_classify_style_detects_single_vs_multi_sentence():
    assert classify_style('החתול ישב על המחצלת') == SINGLE_SENTENCE
    assert classify_style('החתול ישב על המחצלת. הכלב ברח.') == MULTI_SENTENCE


def test_style_summary_counts_and_percentages():
    rows = [{"style_label": PIPE_DIGEST}, {"style_label": PIPE_DIGEST}, {"style_label": SINGLE_SENTENCE}]

    summary = style_summary(rows)

    assert summary[PIPE_DIGEST]["count"] == 2
    assert summary[SINGLE_SENTENCE]["count"] == 1
    assert summary[PIPE_DIGEST]["pct"] == round(2 / 3, 3)
