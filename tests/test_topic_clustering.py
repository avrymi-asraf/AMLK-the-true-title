"""Tests for evaluation.topic_clustering.

The real BERTopic fit + Gemini naming call download an embedding model and hit the API, so
that test is gated behind RUN_LIVE_TESTS, mirroring the live Gemini judge test in
tests/test_evaluation.py.
"""
import os

import pytest


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run the live BERTopic + Gemini naming test",
)
def test_live_cluster_dataset_names_a_topic():
    from evaluation.topic_clustering import NOISE_LABEL, cluster_dataset, plot_clusters

    sports_summaries = [
        "הנבחרת ניצחה במשחק הכדורגל אמש בגמר הליגה",
        "השחקן הבקיע שני שערים במשחק הכדורגל",
        "קבוצת הכדורסל זכתה באליפות אחרי ניצחון דרמטי",
        "הכוכב של הקבוצה נפצע במהלך משחק הכדורגל",
    ]
    # HDBSCAN needs enough points to form a cluster at all — repeat a tiny, clearly-one-topic
    # sample rather than standing up a large synthetic corpus just for this smoke test.
    records = [{"summary": s, "source": "test"} for s in sports_summaries] * 15

    rows, topic_model, embeddings = cluster_dataset(records, min_cluster_size=10)

    assert len(rows) == len(records)
    real_labels = {row["topic_label"] for row in rows if row["topic_label"] != NOISE_LABEL}
    assert real_labels, "expected at least one non-noise cluster to be named"

    fig = plot_clusters(topic_model, [r["summary"] for r in records], embeddings)
    assert fig is not None
