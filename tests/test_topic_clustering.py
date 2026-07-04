"""Tests for evaluation.topic_clustering.

The real BERTopic fit + Gemini naming call download an embedding model and hit the API, so
that test is gated behind RUN_LIVE_TESTS, mirroring the live Gemini judge test in
tests/test_evaluation.py.
"""
import os
import re

import pytest

from evaluation.topic_clustering import HEBREW_STOPWORDS, HEBREW_TOKEN_PATTERN


def test_hebrew_token_pattern_keeps_only_hebrew_words():
    tokens = re.findall(HEBREW_TOKEN_PATTERN, "10 24 ynet nrg כדורגל ניצחון 2013")

    assert tokens == ["כדורגל", "ניצחון"]


def test_hebrew_token_pattern_matches_final_form_letters():
    assert re.findall(HEBREW_TOKEN_PATTERN, "שוק") == ["שוק"]
    assert re.findall(HEBREW_TOKEN_PATTERN, "כלכלן") == ["כלכלן"]


def test_hebrew_stopwords_cover_common_function_words():
    assert {"של", "את", "על", "גם", "לא"} <= HEBREW_STOPWORDS


def test_bertopic_english_preprocess_strips_hebrew():
    """BERTopic's default language='english' removes all Hebrew before c-TF-IDF — fit_topics()
    must pass language='multilingual' instead (see fit_topics docstring)."""
    pytest.importorskip("bertopic")
    import numpy as np
    from bertopic import BERTopic

    docs = np.array(["הנבחרת ניצחה במשחק הכדורגל"])
    english = BERTopic(language="english")._preprocess_text(docs)[0]
    multilingual = BERTopic(language="multilingual")._preprocess_text(docs)[0]

    assert "הנבחרת" not in english
    assert "הנבחרת" in multilingual


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
