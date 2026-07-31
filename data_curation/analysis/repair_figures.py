"""F6 and F7 from `docs/obsidian/Paper Figures.md` — the E2 (graded repair) and E3 (head-to-head)
result figures reported in `paper/main.tex` (Section: Results, "E2/E3: the repair helped").
Reads the two summary artifacts already produced by the E2/E3 judge passes
(`outputs/results/e2_repair_summary.json`, `e3_pairwise_summary.json`) — no rejoining of raw
per-row judge output is needed since both summaries already carry the aggregated numbers this
module plots. Local, CPU-only: Plotly + kaleido for rendering, no GPU/API.

Run:
    python -m data_curation.analysis.repair_figures

Output:
    outputs/figures/f6_paired_repair.{html,svg,png}
    outputs/figures/f6_transition_heatmap.{html,svg,png}
    outputs/figures/f7_win_rate.{html,svg,png}
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_curation.analysis.plotting import CATEGORICAL_PALETTE, apply_house_style, ordinal_palette, save_figure
from data_curation.utils.json_io import load_json

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "results"
E2_SUMMARY_PATH = RESULTS_DIR / "e2_repair_summary.json"
E3_SUMMARY_PATH = RESULTS_DIR / "e3_pairwise_summary.json"

DIMENSIONS = ["faithfulness", "single_focus", "informativeness", "cleanliness"]
DIMENSION_LABELS = {
    "faithfulness": "Faithfulness",
    "single_focus": "Single-focus",
    "informativeness": "Informativeness",
    "cleanliness": "Cleanliness",
}


def build_f6_paired_repair(e2_summary: dict) -> go.Figure:
    """F6a — dumbbell of median original vs.\\ curated score per dimension, over the rewritten rows.

    Every dimension sits at ceiling (5.0) both before and after, so the dumbbell alone looks flat
    by construction (main.tex: "the dumbbell plot alone looks flat") — the share-increased /
    share-decreased annotations are what carry the actual finding on this figure, since the
    Wilcoxon test (not the median) is what detects the ceiling-compressed shift.
    """
    by_dim = e2_summary["by_edit_type"]["all"]
    fig = go.Figure()
    jitter = 0.12  # separates the original/curated markers when medians coincide exactly (ceiling)

    for i, dim in enumerate(DIMENSIONS):
        entry = by_dim[dim]
        y = float(i)
        fig.add_trace(go.Scatter(
            x=[entry["original_median"], entry["curated_median"]], y=[y - jitter, y + jitter],
            mode="lines", line=dict(color="#cccccc", width=3), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[entry["original_median"]], y=[y - jitter], mode="markers", name="Original" if i == 0 else None,
            marker=dict(color=CATEGORICAL_PALETTE[1], size=13), showlegend=(i == 0),
            legendgroup="original",
        ))
        fig.add_trace(go.Scatter(
            x=[entry["curated_median"]], y=[y + jitter], mode="markers", name="Curated" if i == 0 else None,
            marker=dict(color=CATEGORICAL_PALETTE[0], size=13), showlegend=(i == 0),
            legendgroup="curated",
        ))
        annotation = (
            f"+{100 * entry['share_increased']:.0f}% up / "
            f"-{100 * entry['share_decreased']:.0f}% down (n={entry['n']}, p={entry['wilcoxon_pvalue']:.1e})"
        )
        fig.add_annotation(x=5.5, y=y, text=annotation, showarrow=False, font=dict(size=11, color="#6b6b6b"),
                            xanchor="left")

    apply_house_style(
        fig,
        "Curation's paired effect on the rewritten headlines (E2)",
        subtitle=(
            f"Median original vs. curated score per dimension, n={e2_summary['n_paired']} paired rows"
            " -- medians sit at ceiling, the Wilcoxon test on the underlying distribution detects the shift"
        ),
        xaxis_title="rubric score (1-5)",
        source_note="Source: E2 paired rubric judge, outputs/results/e2_repair_summary.json",
        width=1200,
        height=480,
    )
    fig.update_xaxes(range=[0.5, 7.8])
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(DIMENSIONS))),
        ticktext=[DIMENSION_LABELS[d] for d in DIMENSIONS], range=[-0.6, len(DIMENSIONS) - 0.4],
    )
    fig.update_layout(margin=dict(l=160, r=260))
    return fig


def build_f6_transition_heatmap(e2_summary: dict) -> go.Figure:
    """F6b — original score by curated score transition heatmap, for the dimension that moved
    most (faithfulness). Answers what the dumbbell cannot: which rows moved, and did any get
    *worse* (off-diagonal mass below the diagonal)?
    """
    dim = e2_summary["transition_heatmap_dimension"]
    matrix = e2_summary["transition_matrix"]  # matrix[original-1][curated-1]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=[str(i) for i in range(1, 6)], y=[str(i) for i in range(1, 6)],
        colorscale="Blues", text=matrix, texttemplate="%{text}",
        colorbar=dict(title="rows"),
    ))
    apply_house_style(
        fig,
        f"Original vs. curated {DIMENSION_LABELS[dim].lower()} score, rewritten rows only",
        subtitle="Off-diagonal mass below the diagonal would mean curation made a row worse -- it is rare",
        xaxis_title="curated score",
        yaxis_title="original score",
        source_note="Source: E1 (original) + E2 (curated) rubric judge, joined by hesum_id",
        width=650,
        height=560,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def build_f7_win_rate(e3_summary: dict) -> go.Figure:
    """F7 — stacked horizontal win-rate bar (curated-wins / tie / original-wins) for the rewritten
    cohort only (main.tex, Section: E3: Was the repair preferred head to head?). The placebo
    (unchanged-text) cohort is reported as a number in the prose but omitted from this figure to
    keep it to a single, focused bar.
    """
    cohorts = [("rewritten", "Rewritten (n={})".format(e3_summary["rewritten"]["n"]))]

    fig = go.Figure()
    outcomes = [("curated_pct", "Curated wins", CATEGORICAL_PALETTE[0]),
                ("tie_pct", "Tie", "#cccccc"),
                ("original_pct", "Original wins", CATEGORICAL_PALETTE[3])]

    for key, label, color in outcomes:
        fig.add_trace(go.Bar(
            y=[c[1] for c in cohorts],
            x=[e3_summary[c[0]][key] for c in cohorts],
            name=label, orientation="h", marker_color=color,
            text=[f"{e3_summary[c[0]][key]:.1f}%" for c in cohorts],
            textposition="inside",
        ))

    fig.update_layout(barmode="stack")
    apply_house_style(
        fig,
        "Blind pairwise preference: curated vs. original headline (E3)",
        subtitle="Blind judge picks the curated headline over the original in the large majority of rewritten pairs",
        xaxis_title="share of judgments",
        source_note="Source: E3 blind pairwise judge, outputs/results/e3_pairwise_summary.json",
        width=1000,
        height=320,
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_layout(margin=dict(l=260, b=90))
    return fig


def main() -> None:
    e2_summary = load_json(E2_SUMMARY_PATH)
    e3_summary = load_json(E3_SUMMARY_PATH)

    f6a = build_f6_paired_repair(e2_summary)
    print("F6 (paired) saved:", save_figure(f6a, "f6_paired_repair"))

    f6b = build_f6_transition_heatmap(e2_summary)
    print("F6 (transition) saved:", save_figure(f6b, "f6_transition_heatmap"))

    f7 = build_f7_win_rate(e3_summary)
    print("F7 saved:", save_figure(f7, "f7_win_rate"))


if __name__ == "__main__":
    main()
