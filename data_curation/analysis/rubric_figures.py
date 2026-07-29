"""F3, F4, and F5 from `docs/obsidian/Paper Figures.md` — the three E1 result figures reported in
`paper/main.tex` (Section: Results, "E1: the flagged strata are measurably worse" and "Length and
lead bias are continuous"). Reads the E1 judge output plus `row_labels.py`'s per-row artifact
(article length, headline-lead overlap) so the figures and `rubric_results.py`'s Table 1 numbers
can never drift apart. Local, CPU-only: Plotly + kaleido for rendering, no GPU/API.

Run:
    python -m data_curation.analysis.rubric_figures

Output:
    outputs/figures/f3_rubric_distributions.{html,svg,png}
    outputs/figures/f4_effect_sizes.{html,svg,png}
    outputs/figures/f5_length_lead_bias.{html,svg,png}
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_curation.analysis.plotting import CATEGORICAL_PALETTE, apply_house_style, ordinal_palette, save_figure
from data_curation.analysis.rubric_results import (
    DIMENSIONS,
    STRATUM_PREDICATES,
    join_e1_to_strata,
    load_e1_scored_rows,
)
from data_curation.analysis.stats import bootstrap_median_ci, cliffs_delta, bootstrap_cliffs_delta_ci
from data_curation.utils.json_io import load_json
from data_curation.utils.paths import ARTIFACTS_DIR

SOURCE_NOTE_E1 = "Source: E1 rubric judge (gemini-2.5-flash-lite) on original headlines, n=9,958"
STRATA_ORDER = ["S0", "S2", "S3", "S4"]
STRATUM_LABELS = {
    "S0": "S0 clean",
    "S2": "S2 multi-pipe",
    "S3": "S3 multi-item",
    "S4": "S4 rewritten",
}
DIMENSION_LABELS = {
    "faithfulness": "Faithfulness",
    "single_focus": "Single-focus",
    "informativeness": "Informativeness",
    "cleanliness": "Cleanliness",
}
NEGLIGIBLE_BAND = 0.147  # |delta| below this counts as negligible (Cliff 1993 convention)


def build_f3_rubric_distributions(groups: dict[str, dict[str, list[int]]]) -> go.Figure:
    """F3 — stacked 1-5 ordinal distributions, one bar per stratum, faceted by dimension.

    Stacked bars rather than box plots or bar-of-means: the scores are ordinal, so a mean invents
    a scale that does not exist, and stacking exposes ceiling effects (S0 at ~100% level-5)
    directly, which is exactly the weakness E2's paired comparison has to contend with.
    """
    colors = ordinal_palette(5)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[DIMENSION_LABELS[d] for d in DIMENSIONS],
        vertical_spacing=0.16, horizontal_spacing=0.22,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), dim in zip(positions, DIMENSIONS):
        for level in range(1, 6):
            shares = []
            for stratum in STRATA_ORDER:
                values = groups[stratum][dim]
                shares.append(100 * values.count(level) / len(values) if values else 0.0)
            fig.add_trace(
                go.Bar(
                    y=[STRATUM_LABELS[s] for s in STRATA_ORDER],
                    x=shares,
                    orientation="h",
                    name=str(level),
                    marker_color=colors[level - 1],
                    showlegend=(row, col) == (1, 1),
                    legendgroup=str(level),
                ),
                row=row, col=col,
            )

    fig.update_layout(barmode="stack")
    apply_house_style(
        fig,
        "Rubric score distributions by defect stratum",
        subtitle="Share of rows at each 1 (worst) to 5 (best) level; S0 sits at ceiling on every dimension",
        source_note=SOURCE_NOTE_E1,
        width=1300,
        height=700,
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(automargin=True)
    fig.update_layout(legend=dict(title="Score", orientation="h", y=-0.08, x=0.5, xanchor="center"))
    return fig


def build_f4_effect_sizes(groups: dict[str, dict[str, list[int]]]) -> go.Figure:
    """F4 — Cliff's delta forest plot vs.\\ S0, one row per stratum-by-dimension, with 95% bootstrap
    CI whiskers and a shaded negligible-effect band.

    The antidote to p-value theatre: at thousands of rows every E1 comparison is significant, so
    the CI width and the shaded band do the interpretive work a p-value cannot.
    """
    rows, deltas, los, his, colors = [], [], [], [], []
    stratum_colors = {"S2": CATEGORICAL_PALETTE[3], "S3": CATEGORICAL_PALETTE[4], "S4": CATEGORICAL_PALETTE[1]}

    for stratum in ["S2", "S3", "S4"]:
        for dim in DIMENSIONS:
            values = groups[stratum][dim]
            delta = cliffs_delta(values, groups["S0"][dim])
            lo, hi = bootstrap_cliffs_delta_ci(values, groups["S0"][dim])
            rows.append(f"{STRATUM_LABELS[stratum]} — {DIMENSION_LABELS[dim]}")
            deltas.append(delta)
            los.append(delta - lo)
            his.append(hi - delta)
            colors.append(stratum_colors[stratum])

    fig = go.Figure()
    fig.add_vrect(x0=-NEGLIGIBLE_BAND, x1=NEGLIGIBLE_BAND, fillcolor="lightgray", opacity=0.35, line_width=0)
    fig.add_vline(x=0, line_color="#999999", line_width=1)
    fig.add_trace(go.Scatter(
        x=deltas, y=rows[::-1] if False else rows, mode="markers",
        error_x=dict(type="data", symmetric=False, array=his, arrayminus=los, thickness=1.5, width=4),
        marker=dict(color=colors, size=9),
    ))
    fig.update_yaxes(autorange="reversed")
    apply_house_style(
        fig,
        "Effect sizes: Cliff's delta vs. S0 clean",
        subtitle="95% bootstrap CI; shaded band is the negligible-effect range (|delta| < 0.147)",
        xaxis_title="Cliff's delta (negative = worse than S0)",
        source_note=SOURCE_NOTE_E1,
        width=1100,
        height=560,
    )
    fig.update_layout(margin=dict(l=220))
    fig.update_xaxes(range=[-1.05, 0.15])
    return fig


def _quantile_bin_medians(
    x_values: list[float], y_values: list[float], *, n_bins: int = 16,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Equal-count (quantile) bins of `x_values`; returns per-bin (log-x center, median-y, lo, hi)."""
    x_arr = np.asarray(x_values)
    y_arr = np.asarray(y_values)
    order = np.argsort(x_arr)
    x_sorted, y_sorted = x_arr[order], y_arr[order]
    edges = np.array_split(np.arange(len(x_sorted)), n_bins)

    centers, medians, los, his = [], [], [], []
    for idx in edges:
        if len(idx) == 0:
            continue
        bin_x = x_sorted[idx]
        bin_y = y_sorted[idx].tolist()
        centers.append(float(np.median(bin_x)))
        medians.append(float(np.median(bin_y)))
        lo, hi = bootstrap_median_ci(bin_y)
        los.append(lo)
        his.append(hi)
    return centers, medians, los, his


def build_f5_length_lead_bias(
    e1_rows: list[dict], row_labels: list[dict],
) -> go.Figure:
    """F5 — two vertically stacked panels sharing a log-article-length x-axis: rubric sub-scores
    (top, E1-scored rows) and headline-lead overlap (bottom, all 10,000 rows), with a vertical
    rule at the 4,000-token filter threshold.

    Plotting all 10,000 rows in the bottom panel (including ones the token filter removed) is
    what justifies treating length as a continuous covariate rather than a stratum (main.tex,
    Section: Why article length is a covariate, not a stratum).
    """
    row_labels_by_id = {row["hesum_id"]: row for row in row_labels}

    top_x, top_y = {dim: [] for dim in DIMENSIONS}, {dim: [] for dim in DIMENSIONS}
    for e1_row in e1_rows:
        row_label = row_labels_by_id.get(e1_row["hesum_id"])
        if row_label is None:
            continue
        log_tokens = math.log(max(row_label["article_tokens"], 1))
        for dim in DIMENSIONS:
            top_x[dim].append(log_tokens)
            top_y[dim].append(e1_row["scores"][dim]["score"])

    bottom_x = [math.log(max(r["article_tokens"], 1)) for r in row_labels]
    bottom_y = [r["headline_lead_overlap"] for r in row_labels]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=["Rubric sub-scores (E1 population)", "Headline-lead word overlap (all 10,000 rows)"])

    for i, dim in enumerate(DIMENSIONS):
        centers, medians, los, his = _quantile_bin_medians(top_x[dim], top_y[dim])
        color = CATEGORICAL_PALETTE[i]
        fig.add_trace(go.Scatter(x=centers + centers[::-1], y=his + los[::-1], fill="toself",
                                  fillcolor=color, opacity=0.12, line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=centers, y=medians, mode="lines+markers", name=DIMENSION_LABELS[dim],
                                  line=dict(color=color, width=2)), row=1, col=1)

    centers_b, medians_b, los_b, his_b = _quantile_bin_medians(bottom_x, bottom_y)
    fig.add_trace(go.Scatter(x=centers_b + centers_b[::-1], y=his_b + los_b[::-1], fill="toself",
                              fillcolor=CATEGORICAL_PALETTE[0], opacity=0.15, line=dict(width=0),
                              showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=centers_b, y=medians_b, mode="lines+markers", name="Lead overlap",
                              line=dict(color=CATEGORICAL_PALETTE[0], width=2), showlegend=False), row=2, col=1)

    threshold = math.log(4000)
    fig.add_vline(x=threshold, line_dash="dash", line_color="#999999",
                  annotation_text="4,000-token filter", annotation_position="top right", row=1, col=1)
    fig.add_vline(x=threshold, line_dash="dash", line_color="#999999", row=2, col=1)

    apply_house_style(
        fig,
        "Reference quality and lead bias decline continuously with article length",
        subtitle="Equal-count bins of log article tokens; ribbons are 95% bootstrap CI of the binned median",
        source_note=f"{SOURCE_NOTE_E1} + row_labels.json (n=10,000)",
        width=1100,
        height=720,
    )
    fig.update_yaxes(title_text="median rubric score (1-5)", row=1, col=1)
    fig.update_yaxes(title_text="median headline-lead overlap", row=2, col=1)
    fig.update_xaxes(title_text="log(article tokens)", row=2, col=1)
    return fig


def main() -> None:
    e1_rows = load_e1_scored_rows()
    row_labels = load_json(ARTIFACTS_DIR / "row_labels.json")
    row_labels_by_id = {row["hesum_id"]: row for row in row_labels}
    groups = join_e1_to_strata(e1_rows, row_labels_by_id)

    f3 = build_f3_rubric_distributions(groups)
    print("F3 saved:", save_figure(f3, "f3_rubric_distributions"))

    f4 = build_f4_effect_sizes(groups)
    print("F4 saved:", save_figure(f4, "f4_effect_sizes"))

    f5 = build_f5_length_lead_bias(e1_rows, row_labels)
    print("F5 saved:", save_figure(f5, "f5_length_lead_bias"))


if __name__ == "__main__":
    main()
