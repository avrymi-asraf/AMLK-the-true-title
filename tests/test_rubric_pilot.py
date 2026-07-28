"""Tests for data_curation.analysis.rubric_pilot: stratified sampling, distribution/degeneracy
checks, and kappa computation, all on synthetic data — no row_labels.json, no API calls needed.
"""
from data_curation.analysis.rubric_pilot import (
    STRATA,
    is_degenerate,
    score_distribution,
    stratified_sample,
    weighted_kappa,
)


def _row(hesum_id, **overrides):
    row = {
        "hesum_id": hesum_id,
        "reached_model_curation": True,
        "source_label": "usable",
        "headline_action": "kept",
        "multi_pipe": False,
    }
    row.update(overrides)
    return row


FIXTURE_ROWS = [
    _row("1"),  # S0 clean
    _row("2"),  # S0 clean
    _row("3", multi_pipe=True, source_label=None, reached_model_curation=False, headline_action=None),  # S2
    _row("4", source_label="unusable_multiple_independent_items", headline_action=None),  # S3
    _row("5", headline_action="rewritten"),  # S4
    _row("6", headline_action="rewritten"),  # S4
]


def test_stratified_sample_respects_per_stratum_cap():
    sampled = stratified_sample(FIXTURE_ROWS, per_stratum=1, seed=0)

    assert set(sampled.keys()) == set(STRATA.keys())
    assert len(sampled["S0_clean"]) == 1
    assert len(sampled["S4_headline_rewritten"]) == 1


def test_stratified_sample_excludes_anchor_rows(monkeypatch):
    import data_curation.analysis.rubric_pilot as pilot_module

    monkeypatch.setattr(pilot_module, "ANCHOR_HESUM_IDS", frozenset({"1", "2"}))
    sampled = stratified_sample(FIXTURE_ROWS, per_stratum=10, seed=0)

    assert sampled["S0_clean"] == []  # both S0 rows in the fixture were anchors


def test_score_distribution_counts_every_level_including_zero():
    scored = [
        {"scores": {"faithfulness": {"score": 5}}},
        {"scores": {"faithfulness": {"score": 5}}},
        {"scores": {"faithfulness": {"score": 3}}},
    ]
    dist = score_distribution(scored, "faithfulness")

    assert dist == {1: 0, 2: 0, 3: 1, 4: 0, 5: 2}


def test_is_degenerate_flags_a_single_dominant_level():
    assert is_degenerate({1: 0, 2: 0, 3: 0, 4: 1, 5: 19}) is True
    assert is_degenerate({1: 2, 2: 3, 3: 5, 4: 6, 5: 4}) is False
    assert is_degenerate({1: 0, 2: 0, 3: 0, 4: 0, 5: 0}) is True  # empty counts as degenerate


def test_weighted_kappa_is_one_for_identical_sequences():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_weighted_kappa_penalizes_large_disagreements_more_than_small():
    pass_one = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    off_by_one = [1, 2, 3, 4, 5, 2, 3, 4, 5, 1]  # every mismatch is adjacent
    off_by_far = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]  # every mismatch is at the opposite end

    small_gap = weighted_kappa(pass_one, off_by_one)
    large_gap = weighted_kappa(pass_one, off_by_far)

    assert large_gap < small_gap
