"""Generate the two final E1 figures from frozen rubric scores and row labels."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline.common.json_io import load_json
from pipeline.common.paths import DATA_CURATION_ARTIFACTS_DIR
from pipeline.common.plotting import CATEGORICAL_PALETTE, apply_house_style, ordinal_palette, save_figure
from pipeline.common.statistics import bootstrap_cliffs_delta_ci, cliffs_delta
from pipeline.stage_03_reference_experiments.e1_flagged_strata.summarize import (
    DIMENSIONS,
    join_e1_to_strata,
    load_e1_scored_rows,
)


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
NEGLIGIBLE_BAND = 0.147


def build_rubric_distributions(groups: dict[str, dict[str, list[int]]]) -> go.Figure:
    colors = ordinal_palette(5)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[DIMENSION_LABELS[dimension] for dimension in DIMENSIONS],
        vertical_spacing=0.16,
        horizontal_spacing=0.22,
    )
    for (row, column), dimension in zip(((1, 1), (1, 2), (2, 1), (2, 2)), DIMENSIONS):
        for level in range(1, 6):
            shares = []
            for stratum in STRATA_ORDER:
                values = groups[stratum][dimension]
                shares.append(100 * values.count(level) / len(values) if values else 0.0)
            figure.add_trace(
                go.Bar(
                    y=[STRATUM_LABELS[stratum] for stratum in STRATA_ORDER],
                    x=shares,
                    orientation="h",
                    name=str(level),
                    marker_color=colors[level - 1],
                    showlegend=(row, column) == (1, 1),
                    legendgroup=str(level),
                ),
                row=row,
                col=column,
            )
    figure.update_layout(barmode="stack")
    apply_house_style(
        figure,
        "Rubric score distributions by defect stratum",
        subtitle="Share at each ordinal level; S0 sits at ceiling on every dimension",
        source_note="Source: E1 rubric judge (gemini-2.5-flash-lite) on original headlines",
        width=1300,
        height=700,
    )
    figure.update_xaxes(range=[0, 100], ticksuffix="%")
    figure.update_yaxes(automargin=True)
    figure.update_layout(legend={"title": "Score", "orientation": "h", "y": -0.08, "x": 0.5, "xanchor": "center"})
    return figure


def build_effect_sizes(groups: dict[str, dict[str, list[int]]]) -> go.Figure:
    labels: list[str] = []
    deltas: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    colors: list[str] = []
    stratum_colors = {
        "S2": CATEGORICAL_PALETTE[3],
        "S3": CATEGORICAL_PALETTE[4],
        "S4": CATEGORICAL_PALETTE[1],
    }
    for stratum in ("S2", "S3", "S4"):
        for dimension in DIMENSIONS:
            values = groups[stratum][dimension]
            delta = cliffs_delta(values, groups["S0"][dimension])
            low, high = bootstrap_cliffs_delta_ci(values, groups["S0"][dimension])
            labels.append(f"{STRATUM_LABELS[stratum]} — {DIMENSION_LABELS[dimension]}")
            deltas.append(delta)
            lower_errors.append(delta - low)
            upper_errors.append(high - delta)
            colors.append(stratum_colors[stratum])

    figure = go.Figure()
    figure.add_vrect(x0=-NEGLIGIBLE_BAND, x1=NEGLIGIBLE_BAND, fillcolor="lightgray", opacity=0.35, line_width=0)
    figure.add_vline(x=0, line_color="#999999", line_width=1)
    figure.add_trace(go.Scatter(
        x=deltas,
        y=labels,
        mode="markers",
        error_x={"type": "data", "symmetric": False, "array": upper_errors, "arrayminus": lower_errors, "thickness": 1.5, "width": 4},
        marker={"color": colors, "size": 9},
    ))
    figure.update_yaxes(autorange="reversed")
    apply_house_style(
        figure,
        "Effect sizes: Cliff's delta vs. S0 clean",
        subtitle="95% bootstrap CI; shaded band is the negligible-effect range (|delta| < 0.147)",
        xaxis_title="Cliff's delta (negative = worse than S0)",
        source_note="Source: E1 rubric judge (gemini-2.5-flash-lite)",
        width=1100,
        height=560,
    )
    figure.update_layout(margin={"l": 220})
    figure.update_xaxes(range=[-1.05, 0.15])
    return figure


def load_groups() -> dict[str, dict[str, list[int]]]:
    labels = load_json(DATA_CURATION_ARTIFACTS_DIR / "row_labels.json")
    return join_e1_to_strata(load_e1_scored_rows(), {row["hesum_id"]: row for row in labels})


def main() -> None:
    groups = load_groups()
    print("E1 distributions saved:", save_figure(build_rubric_distributions(groups), "e1_rubric_distributions"))
    print("E1 effect sizes saved:", save_figure(build_effect_sizes(groups), "e1_effect_sizes"))


if __name__ == "__main__":
    main()
