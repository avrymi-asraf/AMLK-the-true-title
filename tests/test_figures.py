"""Tests for data_curation.analysis.figures: count derivation and figure construction on a tiny
synthetic row-label fixture. No real artifact, no rendering to disk, no kaleido/network calls.
"""
from data_curation.analysis.figures import (
    build_f1_curation_funnel,
    build_f2_defect_prevalence,
    compute_funnel_counts,
    compute_stratum_counts,
)


def _row(**overrides) -> dict:
    row = {
        "over_token_budget": False,
        "multi_pipe": False,
        "reached_model_curation": True,
        "source_label": "usable",
        "headline_action": "kept",
    }
    row.update(overrides)
    return row


FIXTURE_ROWS = [
    _row(),  # clean, kept
    _row(headline_action="rewritten"),  # usable but rewritten
    _row(multi_pipe=True, reached_model_curation=False, source_label=None, headline_action=None),
    _row(over_token_budget=True, reached_model_curation=False, source_label=None, headline_action=None),
    _row(over_token_budget=True, multi_pipe=True, reached_model_curation=False,
         source_label=None, headline_action=None),  # both filters
    _row(source_label="unusable_multiple_independent_items", headline_action=None),
    _row(source_label="unusable_insufficient_substantive_content", headline_action=None),
]


def test_compute_funnel_counts_partitions_all_rows():
    counts = compute_funnel_counts(FIXTURE_ROWS)

    assert counts["total"] == len(FIXTURE_ROWS)
    assert counts["over_budget_only"] == 1
    assert counts["multi_pipe_only"] == 1
    assert counts["both_filters"] == 1
    assert counts["reached_model_curation"] == 4
    assert counts["usable"] == 2
    assert counts["unusable"] == 2
    assert counts["headline_kept"] == 1
    assert counts["headline_rewritten"] == 1


def test_compute_stratum_counts_matches_taxonomy_labels():
    strata = dict((label, count) for label, count, _color in compute_stratum_counts(FIXTURE_ROWS))

    assert strata["clean (S0)"] == 1
    assert strata["headline_rewritten (S4)"] == 1
    assert strata["multi_pipe_headline (S2)"] == 2  # both the multi-pipe-only and both-filters rows
    assert strata["multiple_independent_items (S3)"] == 1
    assert strata["insufficient_substantive_content (S5)"] == 1


def test_build_f1_curation_funnel_returns_a_sankey_figure():
    fig = build_f1_curation_funnel(FIXTURE_ROWS)

    assert fig.data[0].type == "sankey"
    assert len(fig.data[0].node.label) == 9  # raw + 3 removal reasons + reached + usable/unusable + kept/rewritten


def test_build_f2_defect_prevalence_returns_one_bar_per_stratum():
    fig = build_f2_defect_prevalence(FIXTURE_ROWS)

    assert fig.data[0].type == "bar"
    assert len(fig.data[0].y) == 7
