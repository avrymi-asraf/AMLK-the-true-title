"""Tests for evaluation.stratify_by_topic: pure join/grouping logic, no model/API calls.

Behavioral, not implementation-dictating — exercises the join key contract (prediction
`reference` == topic `summary`) and the skip-small/keep-noise grouping rules from
docs/superpowers/specs/2026-07-04-topic-clustering-design.md.
"""
from evaluation.stratify_by_topic import group_by_topic, join_predictions_with_topics
from evaluation.topic_clustering import NOISE_LABEL


def test_join_matches_on_reference_equals_summary():
    predictions = [{"reference": "סיכום א", "prediction": "תחזית"},
                   {"reference": "סיכום ב", "prediction": "תחזית ב"}]
    topics_by_summary = {"סיכום א": {"topic_label": "ספורט"}}

    joined, n_unmatched = join_predictions_with_topics(predictions, topics_by_summary)

    assert len(joined) == 1
    assert joined[0]["topic_label"] == "ספורט"
    assert n_unmatched == 1


def test_group_by_topic_skips_small_real_topics():
    joined = [{"topic_label": "ספורט"} for _ in range(3)] + [{"topic_label": "כלכלה"} for _ in range(20)]

    groups = group_by_topic(joined, min_count=10)

    assert groups["ספורט"]["skipped"] == "n too small"
    assert "rows" in groups["כלכלה"]
    assert groups["כלכלה"]["n"] == 20


def test_group_by_topic_never_skips_or_merges_noise():
    joined = [{"topic_label": NOISE_LABEL} for _ in range(2)]

    groups = group_by_topic(joined, min_count=10)

    assert "rows" in groups[NOISE_LABEL]
    assert groups[NOISE_LABEL]["n"] == 2
