"""Tests for build_human_validation_sample: stratified draw, anchor exclusion, pairwise subset."""

from data_curation.analysis.build_human_validation_sample import (
    build_worklist_rows,
    build_human_validation_worklist,
    row_strata,
    select_pairwise_ids,
    unique_rubric_ids,
)
from data_curation.analysis.rubric_anchors import ANCHOR_HESUM_IDS
from data_curation.analysis.rubric_pilot import STRATA


def _row(hesum_id, **kwargs):
    base = {
        "hesum_id": hesum_id,
        "reached_model_curation": True,
        "source_label": "usable",
        "headline_action": "kept",
        "multi_pipe": False,
    }
    base.update(kwargs)
    return base


def test_row_strata_clean_and_multi_pipe():
    clean = _row("1", headline_action="kept")
    pipe = _row("2", multi_pipe=True, headline_action="rewritten")
    assert "S0_clean" in row_strata(clean)
    assert "S2_multi_pipe_headline" in row_strata(pipe)
    assert "S4_headline_rewritten" in row_strata(pipe)


def test_unique_rubric_ids_dedupes_overlapping_strata():
    row = _row("10", multi_pipe=True, headline_action="rewritten")
    sampled = {
        "S2_multi_pipe_headline": [row],
        "S4_headline_rewritten": [row],
    }
    entries = unique_rubric_ids(sampled)
    assert len(entries) == 1
    assert set(entries[0]["strata"]) == {"S2_multi_pipe_headline", "S4_headline_rewritten"}


def test_select_pairwise_only_from_rewritten():
    entries = [
        {"hesum_id": "1", "row_label": _row("1", headline_action="kept")},
        {"hesum_id": "2", "row_label": _row("2", headline_action="rewritten")},
        {"hesum_id": "3", "row_label": _row("3", headline_action="rewritten")},
    ]
    chosen = select_pairwise_ids(entries, pairwise_n=1, seed=42)
    assert len(chosen) == 1
    assert chosen <= {"2", "3"}


def test_build_worklist_rows_pairwise_requires_curated_headline():
    entries = [
        {"hesum_id": "5", "strata": ["S4_headline_rewritten"], "row_label": _row("5", headline_action="rewritten")},
    ]
    tail = {"5": {"text": "מאמר", "headline": "כותרת מקורית"}}
    curated = {"5": {"headline": "כותרת מעודכנת"}}
    rows = build_worklist_rows(entries, {"5"}, tail, curated)
    assert rows[0]["tasks"] == ["rubric", "pairwise"]
    assert rows[0]["curated_headline"] == "כותרת מעודכנת"

    rows_no_curated = build_worklist_rows(entries, {"5"}, tail, {})
    assert rows_no_curated[0]["tasks"] == ["rubric"]


def test_build_human_validation_worklist_excludes_anchors(monkeypatch):
    anchor_id = next(iter(ANCHOR_HESUM_IDS))
    labels = [
        _row(anchor_id, headline_action="kept"),
        _row("100", headline_action="kept"),
        _row("101", multi_pipe=True),
    ]
    # pad pools so stratified_sample has enough per stratum
    for i in range(200, 260):
        labels.append(_row(str(i), headline_action="kept"))
    for i in range(300, 360):
        labels.append(_row(str(i), multi_pipe=True))
    for i in range(400, 460):
        labels.append(_row(str(i), source_label="unusable_multiple_independent_items"))
    for i in range(500, 560):
        labels.append(_row(str(i), headline_action="rewritten"))

    def fake_tail(ids):
        return {hid: {"text": f"text {hid}", "headline": f"headline {hid}"} for hid in ids}

    def fake_curated(ids):
        return {hid: {"headline": f"curated {hid}"} for hid in ids}

    monkeypatch.setattr(
        "data_curation.analysis.build_human_validation_sample.load_tail_by_id",
        fake_tail,
    )
    monkeypatch.setattr(
        "data_curation.analysis.build_human_validation_sample.load_curated_by_id",
        fake_curated,
    )

    wl = build_human_validation_worklist(
        n_per_stratum=5,
        pairwise_n=3,
        seed=99,
        row_labels=labels,
    )
    ids = {r["hesum_id"] for r in wl["rows"]}
    assert anchor_id not in ids
    assert wl["seed"] == 99
    assert len(wl["rows"]) <= 20  # 5 per stratum with overlap
    assert all("rubric" in r["tasks"] for r in wl["rows"])


def test_build_human_validation_worklist_deterministic(monkeypatch):
    labels = [_row(str(i), headline_action="kept") for i in range(1, 200)]
    for i in range(200, 250):
        labels.append(_row(str(i), multi_pipe=True, headline_action="rewritten"))

    monkeypatch.setattr(
        "data_curation.analysis.build_human_validation_sample.load_tail_by_id",
        lambda ids: {hid: {"text": "t", "headline": "h"} for hid in ids},
    )
    monkeypatch.setattr(
        "data_curation.analysis.build_human_validation_sample.load_curated_by_id",
        lambda ids: {hid: {"headline": "c"} for hid in ids},
    )

    a = build_human_validation_worklist(n_per_stratum=10, pairwise_n=5, seed=7, row_labels=labels)
    b = build_human_validation_worklist(n_per_stratum=10, pairwise_n=5, seed=7, row_labels=labels)
    assert [r["hesum_id"] for r in a["rows"]] == [r["hesum_id"] for r in b["rows"]]
