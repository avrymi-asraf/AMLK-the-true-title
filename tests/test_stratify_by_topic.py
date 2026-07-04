"""Tests for evaluation.stratify_by_topic: pure join/grouping logic, no model/API calls.

Behavioral, not implementation-dictating — exercises the join key contract (prediction
`reference` == label row `summary`) and the skip-small/never-skip grouping rules from
docs/superpowers/specs/2026-07-04-topic-clustering-design.md. The same generic functions serve
both topic_label (topic_clustering.py) and style_label (style_labels.py) groupings.
"""
from evaluation.stratify_by_topic import group_by_label, join_predictions_with_labels
from evaluation.topic_clustering import NOISE_LABEL


def test_join_matches_on_reference_equals_summary():
    predictions = [{"reference": "סיכום א", "prediction": "תחזית"},
                   {"reference": "סיכום ב", "prediction": "תחזית ב"}]
    rows_by_summary = {"סיכום א": {"topic_label": "ספורט"}}

    joined, n_unmatched = join_predictions_with_labels(predictions, rows_by_summary)

    assert len(joined) == 1
    assert joined[0]["topic_label"] == "ספורט"
    assert n_unmatched == 1


def test_join_works_with_style_label_field():
    predictions = [{"reference": "כותרת א | כותרת ב", "prediction": "תחזית"}]
    rows_by_summary = {"כותרת א | כותרת ב": {"style_label": "pipe_digest"}}

    joined, n_unmatched = join_predictions_with_labels(predictions, rows_by_summary, label_field="style_label")

    assert joined[0]["style_label"] == "pipe_digest"
    assert n_unmatched == 0


def test_group_by_label_skips_small_groups():
    joined = [{"topic_label": "ספורט"} for _ in range(3)] + [{"topic_label": "כלכלה"} for _ in range(20)]

    groups = group_by_label(joined, min_count=10)

    assert groups["ספורט"]["skipped"] == "n too small"
    assert "rows" in groups["כלכלה"]
    assert groups["כלכלה"]["n"] == 20


def test_group_by_label_never_skips_labels_in_never_skip():
    joined = [{"topic_label": NOISE_LABEL} for _ in range(2)]

    groups = group_by_label(joined, min_count=10, never_skip=frozenset({NOISE_LABEL}))

    assert "rows" in groups[NOISE_LABEL]
    assert groups[NOISE_LABEL]["n"] == 2
