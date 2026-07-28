"""Tests for human_validation_results — kappa alignment and pairwise outcome mapping."""

from data_curation.analysis.human_validation_results import (
    human_pairwise_outcome,
    paired_dimension_scores,
    rubric_agreement,
    rubric_scores_by_id,
)


def test_rubric_agreement_perfect_human_human():
    human = {
        "1": {"faithfulness": 4, "single_focus": 3, "informativeness": 4, "cleanliness": 5},
        "2": {"faithfulness": 2, "single_focus": 1, "informativeness": 3, "cleanliness": 4},
    }
    judge = human.copy()
    result = rubric_agreement(human, human, judge)
    for dim in ("faithfulness", "single_focus", "informativeness", "cleanliness"):
        assert result["human_human"][dim] == 1.0
        assert result["judge_human_a"][dim] == 1.0


def test_paired_dimension_scores_aligns_ids():
    a = {"1": {"faithfulness": 4}, "2": {"faithfulness": 5}}
    b = {"2": {"faithfulness": 3}, "3": {"faithfulness": 1}}
    paired = paired_dimension_scores(a, b)
    assert paired["faithfulness"] == ([5], [3])


def test_human_pairwise_outcome_maps_slots():
    rec = {
        "winner": "a",
        "slot_map": {"a": "curated", "b": "original"},
    }
    assert human_pairwise_outcome(rec) == "curated"
    rec2 = {
        "winner": "b",
        "slot_map": {"a": "curated", "b": "original"},
    }
    assert human_pairwise_outcome(rec2) == "original"


def test_rubric_scores_by_id_filters_tasks():
    records = [
        {"hesum_id": "1", "task": "rubric", "scores": {"faithfulness": 4}},
        {"hesum_id": "2", "task": "pairwise", "winner": "a"},
    ]
    assert rubric_scores_by_id(records) == {"1": {"faithfulness": 4}}
