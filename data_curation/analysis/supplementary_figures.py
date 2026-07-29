"""Appendix figures referenced by `paper/main.tex`: filter overlap (`sx01`), headline edit
sub-types (`sx05`), and rubric instrument test-retest agreement (`sx16`). All three read
artifacts that already exist on disk (`row_labels.py`'s per-row artifact and the rubric pilot
summary) rather than any judge call, so they are cheap to (re)build alongside the main-body
figures in `figures.py`, `rubric_figures.py`, and `repair_figures.py`. Local, CPU-only: Plotly +
kaleido for rendering, no GPU/API.

Run:
    python -m data_curation.analysis.supplementary_figures

Output:
    outputs/figures/sx01_filter_overlap.{html,svg,png}
    outputs/figures/sx05_headline_edit_subtypes.{html,svg,png}
    outputs/figures/sx16_pilot_kappa.{html,svg,png}
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import plotly.graph_objects as go

from data_curation.analysis.figures import compute_funnel_counts
from data_curation.analysis.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure
from data_curation.analysis.row_labels import load_row_labels
from data_curation.utils.json_io import load_json

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "results"
RUBRIC_PILOT_PATH = RESULTS_DIR / "rubric_pilot.json"

EDIT_TYPE_LABELS = {
    "pipes_removed": "pipes_removed",
    "boilerplate_stripped": "boilerplate_stripped",
    "truncation_repaired": "truncation_repaired",
    "light_edit": "light_edit",
    "full_rewrite": "full_rewrite",
}
DIMENSION_LABELS = {
    "faithfulness": "Faithfulness",
    "single_focus": "Single-focus",
    "informativeness": "Informativeness",
    "cleanliness": "Cleanliness",
}


def build_sx01_filter_overlap(rows: list[dict]) -> go.Figure:
    """sx01 — bar chart of the two deterministic filters' overlap: over-budget only, multi-pipe
    only, and both, matching the Sankey's own removal-reason breakdown (F1) from a second angle.
    """
    counts = compute_funnel_counts(rows)
    categories = [
        ("Over-budget only", counts["over_budget_only"], CATEGORICAL_PALETTE[1]),
        ("Multi-pipe only", counts["multi_pipe_only"], CATEGORICAL_PALETTE[3]),
        ("Both filters", counts["both_filters"], CATEGORICAL_PALETTE[4]),
    ]
    fig = go.Figure(go.Bar(
        x=[c[0] for c in categories], y=[c[1] for c in categories],
        marker_color=[c[2] for c in categories],
        text=[f"{c[1]:,}" for c in categories], textposition="outside",
    ))
    apply_house_style(
        fig,
        "Overlap between the token-budget and multi-pipe-headline filters",
        subtitle="The two deterministic filters are independent -- most removed rows trip only one of them",
        yaxis_title="rows",
        source_note="Source: row_labels.json (n=10,000)",
        width=800,
        height=500,
    )
    return fig


def build_sx05_headline_edit_subtypes(rows: list[dict]) -> go.Figure:
    """sx05 — distribution of the post-hoc headline edit sub-type among the rewritten rows,
    which is what the E3 win-rate-by-sub-type breakout (main.tex, Section: E2/E3 results) groups by.
    """
    edit_types = [r["headline_edit_type"] for r in rows if r["headline_action"] == "rewritten"]
    counts = Counter(edit_types)
    order = ["full_rewrite", "light_edit", "boilerplate_stripped", "truncation_repaired", "pipes_removed"]
    order = [o for o in order if o in counts] + [o for o in counts if o not in order]

    fig = go.Figure(go.Bar(
        y=[EDIT_TYPE_LABELS.get(o, o) for o in order][::-1],
        x=[counts[o] for o in order][::-1],
        orientation="h", marker_color=CATEGORICAL_PALETTE[0],
        text=[f"{counts[o]:,} ({counts[o] / len(edit_types):.1%})" for o in order][::-1],
        textposition="outside", cliponaxis=False,
    ))
    apply_house_style(
        fig,
        "Distribution of headline edit sub-types among the rewrites",
        subtitle=f"n={len(edit_types):,} rewritten rows, sub-typed post hoc from the (original, replacement) string pair",
        xaxis_title="rows",
        source_note="Source: row_labels.json",
        width=900,
        height=450,
    )
    fig.update_xaxes(range=[0, max(counts.values()) * 1.25])
    fig.update_layout(margin=dict(l=200))
    return fig


def build_sx16_pilot_kappa(pilot: dict) -> go.Figure:
    """sx16 — test-retest quadratically weighted Cohen's kappa per rubric dimension, from the
    300-row pilot (main.tex, Appendix: Rubric instrument validation).
    """
    kappa = pilot["test_retest_kappa"]
    dims = list(DIMENSION_LABELS.keys())
    values = [kappa[d] for d in dims]

    fig = go.Figure(go.Bar(
        x=[DIMENSION_LABELS[d] for d in dims], y=values,
        marker_color=CATEGORICAL_PALETTE[0],
        text=[f"{v:.2f}" for v in values], textposition="outside",
    ))
    fig.add_hline(y=0.6, line_dash="dash", line_color="#999999",
                   annotation_text="substantial agreement threshold", annotation_position="bottom right")
    apply_house_style(
        fig,
        "Rubric test-retest agreement (quadratically weighted kappa)",
        subtitle=f"n={pilot['n_sampled']} pilot rows, {pilot['n_retest']} re-scored at nonzero temperature",
        yaxis_title="quadratically weighted Cohen's kappa",
        source_note="Source: outputs/results/rubric_pilot.json",
        width=800,
        height=500,
    )
    fig.update_yaxes(range=[0, 1.0])
    return fig


def main() -> None:
    rows = load_row_labels()
    pilot = load_json(RUBRIC_PILOT_PATH)

    print("sx01 saved:", save_figure(build_sx01_filter_overlap(rows), "sx01_filter_overlap"))
    print("sx05 saved:", save_figure(build_sx05_headline_edit_subtypes(rows), "sx05_headline_edit_subtypes"))
    print("sx16 saved:", save_figure(build_sx16_pilot_kappa(pilot), "sx16_pilot_kappa"))


if __name__ == "__main__":
    main()
