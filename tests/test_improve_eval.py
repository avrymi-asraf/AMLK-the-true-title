"""Behavior tests for the training-improvement measurement instrument (no API calls)."""
import json

from data.distill import has_foreign_script, is_usable_target
from evaluation.improve_eval import SUBSET_N, load_predictions, paired_delta, subset_indices, take_subset


def test_subset_is_deterministic_and_sorted():
    a = subset_indices(586)
    b = subset_indices(586)
    assert a == b == sorted(a)
    assert len(a) == SUBSET_N
    assert max(a) < 586


def test_subset_returns_all_when_split_is_smaller():
    assert subset_indices(50) == list(range(50))


def test_take_subset_pairs_two_files_on_the_same_rows():
    rows_a = [{"text": f"t{i}", "prediction": "p"} for i in range(586)]
    rows_b = [{"text": f"t{i}", "prediction": "q"} for i in range(586)]
    assert [r["text"] for r in take_subset(rows_a)] == [r["text"] for r in take_subset(rows_b)]


def test_load_predictions_survives_unicode_line_separators(tmp_path):
    # Hebrew article text carries U+2028; str.splitlines() would cut the row in half.
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(
        json.dumps({"text": f"a b{i}", "prediction": "x"}, ensure_ascii=False)
        for i in range(3)
    ), encoding="utf-8")
    assert len(load_predictions(path)) == 3


def test_paired_delta_reports_direction_and_uncertainty():
    arm = {"per_example": [{"faithfulness": 4}, {"faithfulness": 4}, {"faithfulness": 5}]}
    ctrl = {"per_example": [{"faithfulness": 3}, {"faithfulness": 3}, {"faithfulness": 3}]}
    d = paired_delta(arm, ctrl, "faithfulness")
    assert d["n"] == 3
    assert d["delta"] > 0
    assert d["ci95"][0] < d["delta"] < d["ci95"][1]


def test_paired_delta_skips_examples_missing_a_score():
    arm = {"per_example": [{"faithfulness": None}, {"faithfulness": 4}]}
    ctrl = {"per_example": [{"faithfulness": 3}, {"faithfulness": 3}]}
    assert paired_delta(arm, ctrl, "faithfulness")["n"] == 1


def test_foreign_script_detection():
    assert has_foreign_script("ynet דיווח")
    assert has_foreign_script("مرحبا")
    assert not has_foreign_script("ראש הממשלה הודיע על הסכם חדש, 2026.")


def test_usable_target_rejects_digests_and_overlong_targets():
    assert is_usable_target("ראש הממשלה הודיע היום על הסכם חדש עם ירדן בנוגע למים.")
    assert not is_usable_target("כותרת אחת | כותרת שנייה | כותרת שלישית")
    assert not is_usable_target("")
    assert not is_usable_target(" ".join(["מילה"] * 60))
