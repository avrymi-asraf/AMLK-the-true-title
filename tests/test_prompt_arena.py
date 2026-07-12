"""Behavioral tests for the prompt-optimization arena (local scoring + ranking half).

Covers the contracts the loop depends on: a bad candidate set is rejected before a GPU job
is submitted, the compliance signals actually fire on the failure modes the prompts target
(digests, rambling, foreign script), a round's rows group per prompt, and ranking prefers
the judge over ROUGE. No GPU, no API — the sweep job itself is not tested here.
"""
import pytest

from evaluation.prompt_arena import (
    PromptCandidate,
    compliance_metrics,
    compliance_score,
    group_by_prompt,
    leaderboard,
    merge_judge,
    rank,
    read_round,
    score_round,
    validate_candidates,
    write_round,
)

GOOD = "ראש הממשלה הודיע על תקציב חדש לחינוך בסך שני מיליארד שקלים."
DIGEST = "ynet | הארץ | מעריב | ידיעות"
RAMBLE = " ".join(["מילה"] * 80) + "."


def test_validate_rejects_missing_placeholder_and_duplicate_ids():
    problems = validate_candidates([
        PromptCandidate(id="a", template="סכם את הכתבה."),  # no {text}
        PromptCandidate(id="a", template="סכם: {text}"),     # duplicate id
    ])
    assert any("placeholder" in p for p in problems)
    assert any("duplicate" in p for p in problems)


def test_validate_accepts_a_clean_set():
    assert validate_candidates([
        PromptCandidate(id="p1", template="סכם במשפט אחד: {text}"),
        PromptCandidate(id="p2", template="כתוב תקציר קצר: {text}"),
    ]) == []


def test_render_substitutes_the_article():
    rendered = PromptCandidate(id="p1", template="סכם: {text}\nתקציר:").render("כתבה")
    assert "כתבה" in rendered and "{text}" not in rendered


def test_compliance_flags_the_failure_modes_prompts_target():
    assert compliance_metrics(DIGEST)["has_list_markers"] is True
    assert compliance_metrics(RAMBLE)["length_ok"] is False
    assert compliance_metrics("The PM announced a budget.")["hebrew_ok"] is False
    good = compliance_metrics(GOOD)
    assert good["length_ok"] and good["sentences_ok"] and good["hebrew_ok"]
    assert not good["has_list_markers"]


def test_compliance_score_separates_good_from_bad():
    assert compliance_score(GOOD) == 1.0
    assert compliance_score(DIGEST) < 1.0
    assert compliance_score("") < compliance_score(GOOD)


def test_score_round_groups_and_aggregates_per_prompt():
    rows = [
        {"prompt_id": "p1", "prediction": GOOD, "reference": GOOD, "text": "כתבה"},
        {"prompt_id": "p2", "prediction": DIGEST, "reference": GOOD, "text": "כתבה"},
    ]
    assert set(group_by_prompt(rows)) == {"p1", "p2"}
    scores = score_round(rows)
    assert scores["p1"]["compliance"] > scores["p2"]["compliance"]
    assert scores["p1"]["n"] == 1
    assert scores["p1"]["rouge"]["rougeL"] > 0  # Hebrew must not tokenize to nothing


def test_rank_prefers_the_judge_over_rouge():
    scores = {
        "high_rouge": {"compliance": 1.0, "rouge": {"rougeL": 0.9},
                       "judge": {"faithfulness_mean": 2.0, "fluency_mean": 2.0}},
        "high_judge": {"compliance": 1.0, "rouge": {"rougeL": 0.1},
                       "judge": {"faithfulness_mean": 5.0, "fluency_mean": 5.0}},
    }
    assert rank(scores)[0] == "high_judge"


def test_rank_falls_back_to_compliance_when_unjudged():
    scores = {
        "sloppy": {"compliance": 0.25, "rouge": {"rougeL": 0.5}},
        "clean": {"compliance": 1.0, "rouge": {"rougeL": 0.4}},
    }
    assert rank(scores)[0] == "clean"


def test_leaderboard_renders_every_prompt_best_first():
    scores = {
        "p1": {"compliance": 1.0, "mean_words": 12, "mean_sentences": 1.0, "hebrew_ratio": 1.0,
               "list_marker_rate": 0.0, "rouge": {"rougeL": 0.3}},
        "p2": {"compliance": 0.5, "mean_words": 60, "mean_sentences": 4.0, "hebrew_ratio": 0.9,
               "list_marker_rate": 1.0, "rouge": {"rougeL": 0.2}},
    }
    table = leaderboard(scores, [PromptCandidate(id="p1", template="{text}", note="baseline")])
    assert "`p1`" in table and "`p2`" in table
    assert table.index("`p1`") < table.index("`p2`")
    assert "baseline" in table


def test_merge_judge_folds_scores_in_without_mutating():
    scores = {"p1": {"compliance": 1.0}}
    merged = merge_judge(scores, {"p1": {"faithfulness_mean": 4.0, "fluency_mean": 5.0}})
    assert merged["p1"]["judge"]["faithfulness_mean"] == 4.0
    assert "judge" not in scores["p1"]


def test_round_files_roundtrip(tmp_path):
    candidates = [PromptCandidate(id="p1", template="סכם: {text}", note="short")]
    write_round(1, candidates, root=tmp_path)
    assert read_round(1, root=tmp_path) == candidates


def test_write_round_refuses_an_invalid_set(tmp_path):
    with pytest.raises(ValueError):
        write_round(1, [PromptCandidate(id="p1", template="no placeholder")], root=tmp_path)
