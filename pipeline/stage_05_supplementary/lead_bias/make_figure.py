"""Generate the supplementary article-length and lead-bias figure."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline.common.json_io import load_json
from pipeline.common.paths import DATA_CURATION_ARTIFACTS_DIR
from pipeline.common.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure
from pipeline.common.statistics import bootstrap_median_ci
from pipeline.stage_03_reference_experiments.e1_flagged_strata.summarize import (
    DIMENSIONS,
    load_e1_scored_rows,
)


DIMENSION_LABELS = {
    "faithfulness": "Faithfulness",
    "single_focus": "Single-focus",
    "informativeness": "Informativeness",
    "cleanliness": "Cleanliness",
}


def _quantile_bin_medians(
    x_values: list[float],
    y_values: list[float],
    *,
    n_bins: int = 16,
) -> tuple[list[float], list[float], list[float], list[float]]:
    x_array = np.asarray(x_values)
    y_array = np.asarray(y_values)
    order = np.argsort(x_array)
    x_sorted, y_sorted = x_array[order], y_array[order]
    centers: list[float] = []
    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for indices in np.array_split(np.arange(len(x_sorted)), n_bins):
        if len(indices) == 0:
            continue
        bin_values = y_sorted[indices].tolist()
        centers.append(float(np.median(x_sorted[indices])))
        medians.append(float(np.median(bin_values)))
        low, high = bootstrap_median_ci(bin_values)
        lows.append(low)
        highs.append(high)
    return centers, medians, lows, highs


def build_lead_bias_figure(e1_rows: list[dict], row_labels: list[dict]) -> go.Figure:
    labels_by_id = {row["hesum_id"]: row for row in row_labels}
    top_x = {dimension: [] for dimension in DIMENSIONS}
    top_y = {dimension: [] for dimension in DIMENSIONS}
    for score_row in e1_rows:
        label = labels_by_id.get(score_row["hesum_id"])
        if label is None:
            continue
        log_tokens = math.log(max(label["article_tokens"], 1))
        for dimension in DIMENSIONS:
            top_x[dimension].append(log_tokens)
            top_y[dimension].append(score_row["scores"][dimension]["score"])

    bottom_x = [math.log(max(row["article_tokens"], 1)) for row in row_labels]
    bottom_y = [row["headline_lead_overlap"] for row in row_labels]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Rubric sub-scores (E1 population)", "Headline-lead word overlap (all rows)"),
    )
    for index, dimension in enumerate(DIMENSIONS):
        centers, medians, lows, highs = _quantile_bin_medians(top_x[dimension], top_y[dimension])
        color = CATEGORICAL_PALETTE[index]
        figure.add_trace(go.Scatter(
            x=centers + centers[::-1],
            y=highs + lows[::-1],
            fill="toself",
            fillcolor=color,
            opacity=0.12,
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)
        figure.add_trace(go.Scatter(
            x=centers,
            y=medians,
            mode="lines+markers",
            name=DIMENSION_LABELS[dimension],
            line={"color": color, "width": 2},
        ), row=1, col=1)

    centers, medians, lows, highs = _quantile_bin_medians(bottom_x, bottom_y)
    figure.add_trace(go.Scatter(
        x=centers + centers[::-1],
        y=highs + lows[::-1],
        fill="toself",
        fillcolor=CATEGORICAL_PALETTE[0],
        opacity=0.15,
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=centers,
        y=medians,
        mode="lines+markers",
        name="Lead overlap",
        line={"color": CATEGORICAL_PALETTE[0], "width": 2},
        showlegend=False,
    ), row=2, col=1)

    threshold = math.log(4000)
    figure.add_vline(x=threshold, line_dash="dash", line_color="#999999", annotation_text="4,000-token filter", annotation_position="top right", row=1, col=1)
    figure.add_vline(x=threshold, line_dash="dash", line_color="#999999", row=2, col=1)
    apply_house_style(
        figure,
        "Reference quality and lead bias decline continuously with article length",
        subtitle="Equal-count bins of log article tokens; ribbons are 95% bootstrap CIs of binned medians",
        source_note="Source: frozen E1 rubric scores and row_labels.json",
        width=1100,
        height=720,
    )
    figure.update_yaxes(title_text="median rubric score (1-5)", row=1, col=1)
    figure.update_yaxes(title_text="median headline-lead overlap", row=2, col=1)
    figure.update_xaxes(title_text="log(article tokens)", row=2, col=1)
    return figure


def main() -> None:
    labels = load_json(DATA_CURATION_ARTIFACTS_DIR / "row_labels.json")
    figure = build_lead_bias_figure(load_e1_scored_rows(), labels)
    print("Lead-bias figure saved:", save_figure(figure, "lead_bias"))


if __name__ == "__main__":
    main()
