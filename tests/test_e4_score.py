"""Offline tests for scripts.e4_score: pairing, cache keys, four-dimension aggregation.

No live Gemini calls — score_headline is mocked where the scoring loop is exercised.
"""
from __future__ import annotations

from pathlib import Path

from evaluation.rubric_judge import DIMENSIONS
from scripts.e4_score import (
    cache_key,
    decision_flag,
    pair_by_text,
    rubric_paired_summary,
    score_arm_rubric,
)


def test_pair_by_text_joins_on_article():
    raw = [
        {"text": "article-a", "prediction": "raw-a"},
        {"text": "article-b", "prediction": "raw-b"},
    ]
    cur = [
        {"text": "article-b", "prediction": "cur-b", "reference": "ref-b"},
        {"text": "article-a", "prediction": "cur-a", "reference": "ref-a"},
        {"text": "article-only-cur", "prediction": "orphan"},
    ]
    pairs = pair_by_text(raw, cur)
    assert len(pairs) == 2
    by_text = {p["text"]: p for p in pairs}
    assert by_text["article-a"]["raw_prediction"] == "raw-a"
    assert by_text["article-a"]["curated_prediction"] == "cur-a"
    assert by_text["article-a"]["reference"] == "ref-a"


def test_cache_key_stable_and_sensitive():
    k1 = cache_key("art", "pred")
    k2 = cache_key("art", "pred")
    k3 = cache_key("art", "other")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64


def _scores(faith: int, focus: int = 5, info: int = 4, clean: int = 5) -> dict:
    return {
        "faithfulness": faith,
        "single_focus": focus,
        "informativeness": info,
        "cleanliness": clean,
    }


def test_rubric_paired_summary_means_and_delta():
    raw = [_scores(3), _scores(3), _scores(4), _scores(2)]
    cur = [_scores(4), _scores(5), _scores(4), _scores(3)]
    summary = rubric_paired_summary(raw, cur)

    assert summary["n_pairs_attempted"] == 4
    assert summary["n_faithfulness_pairs"] == 4
    assert summary["raw_faithfulness_mean"] == 3.0
    assert summary["curated_faithfulness_mean"] == 4.0
    assert summary["faithfulness_mean_delta_curated_minus_raw"] == 1.0
    # curated higher → positive Cliff's δ
    assert summary["faithfulness_cliffs_delta_curated_vs_raw"] > 0
    assert set(summary["by_dimension"]) == set(DIMENSIONS)


def test_rubric_paired_summary_skips_partial_dims():
    raw = [
        _scores(3),
        {"faithfulness": 2},  # incomplete: only faithfulness
    ]
    cur = [
        _scores(5),
        {"single_focus": 1},  # incomplete: only single_focus — no faithfulness pair
    ]
    summary = rubric_paired_summary(raw, cur)
    # only row 0 has faithfulness on both arms
    assert summary["n_faithfulness_pairs"] == 1
    assert summary["faithfulness_cliffs_delta_curated_vs_raw"] is None
    assert summary["faithfulness_ci_excludes_0"] is False
    # only row 0 has single_focus on both arms (row 1 raw lacks it)
    assert summary["n_single_focus_pairs"] == 1
    assert summary["single_focus_cliffs_delta_curated_vs_raw"] is None


def test_score_arm_rubric_uses_cache_without_api(tmp_path: Path):
    rows = [
        {"text": "article one", "prediction": "summary one"},
        {"text": "article two", "prediction": "summary two"},
    ]
    cache_path = tmp_path / "cache.json"
    preloaded = {
        cache_key(rows[0]["text"], rows[0]["prediction"]): _scores(5, 5, 5, 5),
        cache_key(rows[1]["text"], rows[1]["prediction"]): _scores(2, 2, 2, 2),
    }
    # model=None would call Gemini on a miss; cache hits must not call anything.
    results = score_arm_rubric(
        rows,
        model=None,
        cache=preloaded,
        cache_path=cache_path,
        arm_label="test",
    )
    assert results[0]["faithfulness"] == 5
    assert results[1]["faithfulness"] == 2


def test_score_arm_rubric_writes_cache_on_hit_from_mock(tmp_path: Path, monkeypatch):
    import scripts.e4_score as mod

    calls: list[tuple[str, str]] = []

    def fake_score(article, headline, model=None, temperature=None):
        calls.append((article, headline))
        return {
            dim: {"score": 4, "justification": "ok"} for dim in DIMENSIONS
        }

    monkeypatch.setattr(mod, "score_headline", fake_score)
    rows = [{"text": "a", "prediction": "p"}]
    cache: dict = {}
    cache_path = tmp_path / "c.json"
    results = mod.score_arm_rubric(
        rows, model="unused", cache=cache, cache_path=cache_path, arm_label="m",
    )
    assert results[0]["faithfulness"] == 4
    assert len(calls) == 1
    assert cache_path.exists()
    # Second pass is cache-only.
    results2 = mod.score_arm_rubric(
        rows, model="unused", cache=cache, cache_path=cache_path, arm_label="m",
    )
    assert results2[0]["faithfulness"] == 4
    assert len(calls) == 1


def test_decision_flag_pairwise_or_faithfulness():
    assert decision_flag({"pairwise": {"excludes_50": True}, "pointwise": {}}) is True
    assert decision_flag({
        "pairwise": {"excludes_50": False},
        "pointwise": {"faithfulness_ci_excludes_0": True},
    }) is True
    assert decision_flag({
        "pairwise": {"excludes_50": False},
        "pointwise": {
            "faithfulness_ci_excludes_0": False,
            "cleanliness_ci_excludes_0": True,
        },
    }) is False
