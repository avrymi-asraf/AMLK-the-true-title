"""Tests for data.clean: pure regex reference normalization, no model/API/GPU.

Behaviour covered: pipe/bullet digests become natural prose, leading list markers are dropped,
the result ends with a terminal period, cleaning is idempotent, and the roundup-digest filter
trips only on genuine multi-headline references. Also covers Hub/repo naming helpers.
"""
from data.clean import is_roundup_digest, normalize_summary, pipe_segments


def test_normalize_summary_rewrites_pipes_to_sentences():
    result = normalize_summary("כותרת אחת | כותרת שנייה | כותרת שלישית")
    assert "|" not in result
    assert result == "כותרת אחת. כותרת שנייה. כותרת שלישית."


def test_normalize_summary_rewrites_bullets():
    result = normalize_summary("• סעיף ראשון • סעיף שני")
    assert "•" not in result
    assert result == "סעיף ראשון. סעיף שני."


def test_normalize_summary_drops_leading_list_marker():
    assert normalize_summary("- פריט בודד").startswith("פריט")


def test_normalize_summary_ensures_terminal_period():
    assert normalize_summary("משפט ללא נקודה").endswith(".")
    # An existing terminal punctuation mark is preserved, not doubled.
    assert normalize_summary("שאלה פתוחה?").endswith("?")
    assert not normalize_summary("שאלה פתוחה?").endswith("?.")


def test_normalize_summary_is_idempotent_on_clean_prose():
    clean = "החתול ישב על המחצלת. הכלב ברח."
    assert normalize_summary(clean) == clean


def test_normalize_summary_handles_empty():
    assert normalize_summary("") == ""


def test_pipe_segments_counts_nonempty_segments():
    assert pipe_segments("א | ב | ג") == 3
    assert pipe_segments("משפט אחד ללא פייפ") == 1


def test_is_roundup_digest_trips_only_on_multi_headline():
    assert is_roundup_digest("א | ב | ג") is True
    assert is_roundup_digest("כותרת אחת | כותרת שנייה") is False  # 2 segments < default threshold 3
    assert is_roundup_digest("סיכום רגיל של כתבה אחת") is False


def test_processed_profile_names():
    from training.config import dataset_repo, model_repo, processed_profile_name

    assert processed_profile_name("whole") == "whole"
    assert processed_profile_name("lead") == "lead"
    assert dataset_repo("user", "whole") == "user/amlk-training-data"
    assert dataset_repo("user", "lead") == "user/amlk-training-data-lead"
    assert model_repo("user", "whole") == "user/amlk-dictalm2-instruct-sft"
    assert model_repo("user", "lead") == "user/amlk-dictalm2-instruct-sft-lead"


def test_wandb_naming_includes_date_model_epochs():
    from training.config import DEFAULT_EPOCHS, wandb_project, wandb_run_name

    assert wandb_project("dictalm2-instruct") == "amlk-dictalm2-instruct"
    name = wandb_run_name(
        "qlora", "whole", model_slug="dictalm2-instruct",
        epochs=DEFAULT_EPOCHS, tag="smoke", run_date="2026-07-11",
    )
    assert name == "2026-07-11_dictalm2-instruct_qlora_whole_1ep_smoke"
    assert DEFAULT_EPOCHS == 1


def test_build_prompt_states_a_word_cap():
    from data.prompts import build_prompt

    result = build_prompt("המאמר")
    assert "מילים" in result  # states a numeric word cap (prompt-arena round-2 winner)
    assert "המאמר" in result
    assert "תקציר" in result
