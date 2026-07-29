"""Tests for extend_human_validation_pairwise."""

from data_curation.analysis.extend_human_validation_pairwise import (
    build_pairwise_extension_rows,
    eligible_rewritten_pool,
    extend_worklist_pairwise,
    sample_extra_rows,
)


def _row(hesum_id, **kwargs):
    base = {
        "hesum_id": hesum_id,
        "reached_model_curation": True,
        "source_label": "usable",
        "headline_action": "rewritten",
        "multi_pipe": False,
    }
    base.update(kwargs)
    return base


def test_eligible_pool_excludes_worklist_ids():
    worklist = {"rows": [{"hesum_id": "1"}]}
    labels = [_row("1"), _row("2"), _row("3", headline_action="kept")]
    pool = eligible_rewritten_pool(worklist, labels)
    assert [r["hesum_id"] for r in pool] == ["2"]


def test_build_pairwise_extension_rows():
    labels = [_row("99")]
    tail = {"99": {"text": "article", "headline": "orig"}}
    curated = {"99": {"headline": "curated"}}
    extra = build_pairwise_extension_rows(
        labels,
        annotator_id="amit",
        tail_by_id=tail,
        curated_by_id=curated,
    )
    assert len(extra) == 1
    assert extra[0]["tasks"] == ["pairwise"]
    assert extra[0]["assigned_annotator"] == "amit"
    assert extra[0]["extension"] == "pairwise_extra"


def test_extend_worklist_pairwise(monkeypatch):
    worklist = {
        "version": "v1-split",
        "rows": [{
            "hesum_id": "10",
            "tasks": ["rubric"],
            "strata": ["S4_headline_rewritten"],
            "assigned_annotator": "amit",
        }],
    }
    labels = [_row("99")]

    monkeypatch.setattr(
        "data_curation.analysis.extend_human_validation_pairwise.load_tail_by_id",
        lambda ids: {i: {"text": "t", "headline": "o"} for i in ids},
    )
    monkeypatch.setattr(
        "data_curation.analysis.extend_human_validation_pairwise.load_curated_by_id",
        lambda ids: {i: {"headline": "c"} for i in ids},
    )

    updated = extend_worklist_pairwise(
        worklist,
        annotator_id="amit",
        count=1,
        seed=1,
        row_labels=labels,
    )
    assert updated["n_rows"] == 2
    assert sum(1 for r in updated["rows"] if r["tasks"] == ["pairwise"]) == 1
    assert updated["assignment"]["amit"] == 2


def test_sample_extra_rows_deterministic():
    pool = [_row(str(i)) for i in range(10)]
    a = sample_extra_rows(pool, 3, seed=7)
    b = sample_extra_rows(pool, 3, seed=7)
    assert [r["hesum_id"] for r in a] == [r["hesum_id"] for r in b]
