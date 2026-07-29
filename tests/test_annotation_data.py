"""Tests for evaluation.viewer.annotation_data — resume, pairwise slots, append/read."""

import json

from evaluation.viewer.annotation_data import (
    append_annotation,
    build_pairwise_record,
    build_rubric_record,
    completed_keys,
    default_annotations_path,
    dedupe_annotations,
    expand_tasks,
    export_summary,
    filter_task_items,
    load_annotations,
    pairwise_presentation,
    upsert_annotation,
)


def test_pairwise_presentation_deterministic_and_differs_by_annotator():
    a1 = pairwise_presentation("alice", "42", "orig", "cur")
    a2 = pairwise_presentation("alice", "42", "orig", "cur")
    b1 = pairwise_presentation("bob", "42", "orig", "cur")

    assert a1 == a2
    assert a1["slot_map"] in ({"a": "original", "b": "curated"}, {"a": "curated", "b": "original"})
    # different annotators may get different slot order
    assert isinstance(b1["curated_is_a"], bool)


def test_completed_keys_and_append_round_trip(tmp_path):
    path = tmp_path / "ann.jsonl"
    rec = build_rubric_record("amit", "1", {
        "faithfulness": 4,
        "single_focus": 3,
        "informativeness": 4,
        "cleanliness": 5,
    })
    append_annotation(path, rec)
    loaded = load_annotations(path)
    assert len(loaded) == 1
    assert completed_keys(loaded) == {("1", "rubric")}


def test_build_pairwise_record():
    rec = build_pairwise_record("x", "9", "a", {"a": "original", "b": "curated"})
    assert rec["winner"] == "a"
    assert rec["task"] == "pairwise"


def test_expand_tasks_filters_by_annotator():
    worklist = {
        "rows": [
            {"hesum_id": "1", "tasks": ["rubric"], "assigned_annotator": "amit"},
            {"hesum_id": "2", "tasks": ["rubric", "pairwise"], "assigned_annotator": "ofek"},
        ]
    }
    amit_items = expand_tasks(worklist, annotator_id="amit")
    assert len(amit_items) == 1
    assert amit_items[0]["hesum_id"] == "1"
    summary = export_summary([], worklist, annotator_id="ofek")
    assert summary["rubric_total"] == 1
    assert summary["pairwise_total"] == 1


def test_expand_tasks_and_export_summary():
    worklist = {
        "rows": [
            {"hesum_id": "1", "tasks": ["rubric"]},
            {"hesum_id": "2", "tasks": ["rubric", "pairwise"]},
        ]
    }
    items = expand_tasks(worklist)
    assert len(items) == 3
    summary = export_summary([], worklist)
    assert summary["rubric_total"] == 2
    assert summary["pairwise_total"] == 1


def test_filter_task_items_remaining():
    items = expand_tasks({
        "rows": [{"hesum_id": "1", "tasks": ["rubric", "pairwise"]}],
    })
    completed = {("1", "rubric")}
    remaining = filter_task_items(items, completed, only_remaining=True)
    assert len(remaining) == 1
    assert remaining[0]["task"] == "pairwise"


def test_upsert_replaces_existing_record(tmp_path):
    path = tmp_path / "ann.jsonl"
    rec1 = build_rubric_record("amit", "1", {
        "faithfulness": 3, "single_focus": 3, "informativeness": 3, "cleanliness": 1,
    })
    upsert_annotation(path, rec1)
    rec2 = build_rubric_record("amit", "1", {
        "faithfulness": 3, "single_focus": 3, "informativeness": 3, "cleanliness": 4,
    })
    upsert_annotation(path, rec2)
    loaded = load_annotations(path)
    assert len(loaded) == 1
    assert loaded[0]["scores"]["cleanliness"] == 4


def test_dedupe_keeps_latest_submitted_at():
    older = build_rubric_record("amit", "1", {
        "faithfulness": 2, "single_focus": 2, "informativeness": 2, "cleanliness": 2,
    })
    newer = build_rubric_record("amit", "1", {
        "faithfulness": 5, "single_focus": 5, "informativeness": 5, "cleanliness": 5,
    })
    newer["submitted_at"] = "2026-07-29T00:00:00+00:00"
    older["submitted_at"] = "2026-07-28T00:00:00+00:00"
    out = dedupe_annotations([older, newer])
    assert len(out) == 1
    assert out[0]["scores"]["faithfulness"] == 5


def test_default_annotations_path_sanitizes():
    path = default_annotations_path("amit.ben")
    assert "human_annotations/amit_ben.jsonl" in str(path)
    assert "artifacts" in str(path)
