"""Tests for evaluation.topic_clustering.

The real BERTopic fit + Gemini naming call download an embedding model and hit the API, so
that test is gated behind RUN_LIVE_TESTS, mirroring the live Gemini judge test in
tests/test_evaluation.py.
"""
import os
import re

import pytest

from evaluation.topic_clustering import (
    HEBREW_STOPWORDS,
    HEBREW_TOKEN_PATTERN,
    NOISE_TOPIC_ID,
    _large_topic_ids,
    _resolve_embed_device,
    _truncate_text,
    merge_duplicate_labels,
    plot_topic_sizes,
    renumber_rows,
    topic_summary,
)


def test_resolve_embed_device_cpu():
    assert _resolve_embed_device("cpu") == "cpu"


def test_resolve_embed_device_rejects_unknown():
    import pytest

    with pytest.raises(ValueError, match="embed_device"):
        _resolve_embed_device("tpu")


def test_large_topic_ids_selects_clusters_above_fraction():
    cluster_ids = [0] * 80 + [1] * 15 + [2] * 5

    assert _large_topic_ids(cluster_ids, size_fraction=0.3) == [0]
    assert _large_topic_ids(cluster_ids, size_fraction=0.2) == [0]
    assert _large_topic_ids(cluster_ids, size_fraction=0.85) == []


def test_large_topic_ids_ignores_noise():
    cluster_ids = [-1] * 50 + [0] * 50

    assert _large_topic_ids(cluster_ids, size_fraction=0.3) == [0]


def test_large_topic_ids_empty_input():
    assert _large_topic_ids([], size_fraction=0.3) == []


def test_renumber_rows_maps_to_contiguous_ids():
    rows = [
        {"cluster_id": 7, "topic_label": "א"},
        {"cluster_id": 0, "topic_label": "ב"},
        {"cluster_id": 7, "topic_label": "א"},
        {"cluster_id": 12, "topic_label": "ג"},
    ]

    renumbered = renumber_rows(rows)

    assert [r["cluster_id"] for r in renumbered] == [1, 0, 1, 2]
    assert {r["topic_label"] for r in renumbered} == {"א", "ב", "ג"}


def test_renumber_rows_keeps_noise_as_minus_one():
    rows = [{"cluster_id": NOISE_TOPIC_ID, "topic_label": "לא מסווג"}, {"cluster_id": 5, "topic_label": "ספורט"}]

    renumbered = renumber_rows(rows)

    assert renumbered[0]["cluster_id"] == NOISE_TOPIC_ID
    assert renumbered[1]["cluster_id"] == 0


def test_truncate_text_limits_article_body():
    body = "א" * 10_000
    assert len(_truncate_text(body, max_chars=4000)) == 4000


def test_merge_duplicate_labels_collapses_same_label_clusters():
    rows = [
        {"summary": "a", "source": "s", "cluster_id": 3, "topic_label": "ספורט", "keywords": ["כדורגל"]},
        {"summary": "b", "source": "s", "cluster_id": 7, "topic_label": "ספורט", "keywords": ["כדורסל"]},
        {"summary": "c", "source": "s", "cluster_id": 1, "topic_label": "פוליטיקה", "keywords": ["ממשלה"]},
    ]

    merged = merge_duplicate_labels(rows)

    sport_ids = {row["cluster_id"] for row in merged if row["topic_label"] == "ספורט"}
    assert sport_ids == {3}  # collapsed onto the smallest original cluster_id
    sport_keywords = next(row["keywords"] for row in merged if row["topic_label"] == "ספורט")
    assert sport_keywords == ["כדורגל", "כדורסל"]  # union, order-preserving, deduped
    assert {row["cluster_id"] for row in merged if row["topic_label"] == "פוליטיקה"} == {1}


def test_merge_duplicate_labels_reduces_topic_summary_row_count():
    rows = [
        {"summary": "a", "source": "s", "cluster_id": 0, "topic_label": "ביטחון", "keywords": []},
        {"summary": "b", "source": "s", "cluster_id": 5, "topic_label": "ביטחון", "keywords": []},
    ]

    merged = merge_duplicate_labels(rows)

    assert len(topic_summary(merged)) == 1
    assert topic_summary(merged)[0]["count"] == 2


def test_merge_duplicate_labels_handles_empty_input():
    assert merge_duplicate_labels([]) == []


def test_plot_topic_sizes_returns_a_figure_with_a_bar_per_topic():
    summary_rows = [
        {"cluster_id": 0, "topic_label": "ספורט", "keywords": [], "count": 100},
        {"cluster_id": 1, "topic_label": "פוליטיקה", "keywords": [], "count": 50},
    ]

    fig = plot_topic_sizes(summary_rows)

    assert fig is not None
    assert len(fig.data[0].x) == 2


def test_plot_topic_sizes_respects_top_n():
    summary_rows = [
        {"cluster_id": i, "topic_label": f"topic{i}", "keywords": [], "count": 100 - i}
        for i in range(10)
    ]

    fig = plot_topic_sizes(summary_rows, top_n=3)

    assert len(fig.data[0].x) == 3


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


def test_bertopic_reduce_topics_uses_topics_on_model_not_second_arg():
    """BERTopic >=0.17 reads assignments from topic_model.topics_, not a topics= arg."""
    pytest.importorskip("bertopic")
    import inspect
    from bertopic import BERTopic

    params = inspect.signature(BERTopic.reduce_topics).parameters
    assert "nr_topics" in params
    assert "topics" not in params


def test_distinct_topic_colors_are_unique():
    from evaluation.topic_clustering import _distinct_topic_colors

    colors = _distinct_topic_colors(12)
    assert len(colors) == 12
    assert len(set(colors)) == 12


def test_spread_display_layout_increases_centroid_distance():
    pytest.importorskip("numpy")
    import numpy as np

    from evaluation.topic_clustering import _spread_display_layout

    emb = np.array([[0.09, 0.0], [0.11, 0.0], [0.135, 0.0], [0.155, 0.0]])
    topics = [0, 0, 1, 1]
    topic_ids = [0, 1]
    spread = _spread_display_layout(emb, topics, topic_ids, strength=1.0)
    d_before = np.linalg.norm(emb[:2].mean(axis=0) - emb[2:].mean(axis=0))
    d_after = np.linalg.norm(spread[:2].mean(axis=0) - spread[2:].mean(axis=0))
    assert d_after > d_before


def test_plot_clusters_3d_uses_scatter3d():
    pytest.importorskip("umap")
    import numpy as np
    from unittest.mock import MagicMock

    from evaluation.topic_clustering import plot_clusters

    topic_model = MagicMock()
    topic_model.topics_ = [0] * 30 + [1] * 30
    topic_model.custom_labels_ = ["נושא א", "נושא ב"]
    rng = np.random.default_rng(0)
    embeddings = np.concatenate([
        rng.normal(scale=0.3, size=(30, 16)),
        rng.normal(scale=0.3, size=(30, 16)) + 4.0,
    ])
    hover = [f"doc{i}" for i in range(60)]

    fig = plot_clusters(topic_model, hover, embeddings, sample=1.0, dimensions=3)

    assert fig is not None
    assert fig.layout.scene is not None
    trace_names = {type(t).__name__ for t in fig.data}
    assert "Scatter3d" in trace_names
    header_text = " ".join(
        t.text[0] for t in fig.data if type(t).__name__ == "Scatter3d" and t.mode == "text"
    )
    assert "נושא א" in header_text
    assert "articles" in header_text


def test_plot_clusters_adds_topic_header_annotations():
    pytest.importorskip("umap")
    import numpy as np
    from unittest.mock import MagicMock

    from evaluation.topic_clustering import plot_clusters

    topic_model = MagicMock()
    topic_model.topics_ = [0] * 30 + [1] * 30
    topic_model.custom_labels_ = ["נושא א", "נושא ב"]
    rng = np.random.default_rng(0)
    embeddings = np.concatenate([
        rng.normal(scale=0.3, size=(30, 16)),
        rng.normal(scale=0.3, size=(30, 16)) + 4.0,
    ])
    hover = [f"doc{i}" for i in range(60)]

    fig = plot_clusters(topic_model, hover, embeddings, sample=1.0)

    assert fig is not None
    assert len(fig.layout.annotations) == 2
    header_text = " ".join(a["text"] for a in fig.layout.annotations)
    assert "נושא א" in header_text
    assert "articles" in header_text


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
    records = [{"summary": s, "text": s, "source": "test"} for s in sports_summaries] * 15

    rows, topic_model, embeddings = cluster_dataset(records, min_cluster_size=10, embed_field="summary")

    assert len(rows) == len(records)
    real_labels = {row["topic_label"] for row in rows if row["topic_label"] != NOISE_LABEL}
    assert real_labels, "expected at least one non-noise cluster to be named"

    fig = plot_clusters(topic_model, [r["summary"] for r in records], embeddings)
    assert fig is not None
