"""Generate the final E2 faithfulness transition heatmap."""

from __future__ import annotations

import plotly.graph_objects as go

from pipeline.common.json_io import load_json
from pipeline.common.paths import REFERENCE_EXPERIMENT_ARTIFACTS_DIR
from pipeline.common.plotting import apply_house_style, save_figure


SUMMARY_PATH = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "e2_repair_summary.json"
DIMENSION_LABELS = {
    "faithfulness": "Faithfulness",
    "single_focus": "Single-focus",
    "informativeness": "Informativeness",
    "cleanliness": "Cleanliness",
}


def build_transition_heatmap(summary: dict) -> go.Figure:
    dimension = summary["transition_heatmap_dimension"]
    matrix = summary["transition_matrix"]
    if dimension not in DIMENSION_LABELS:
        raise ValueError(f"unknown E2 transition dimension: {dimension!r}")
    if len(matrix) != 5 or any(not isinstance(row, list) or len(row) != 5 for row in matrix):
        raise ValueError("E2 transition_matrix must be a 5x5 list")
    figure = go.Figure(go.Heatmap(
        z=matrix,
        x=[str(level) for level in range(1, 6)],
        y=[str(level) for level in range(1, 6)],
        colorscale="Blues",
        text=matrix,
        texttemplate="%{text}",
        colorbar={"title": "rows"},
    ))
    apply_house_style(
        figure,
        f"Original vs. curated {DIMENSION_LABELS[dimension].lower()} score, rewritten rows only",
        subtitle="Off-diagonal mass below the diagonal marks rows whose curated target scored lower",
        xaxis_title="curated score",
        yaxis_title="original score",
        source_note="Source: paired E1 original and E2 curated rubric judgments",
        width=650,
        height=560,
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def main() -> None:
    summary = load_json(SUMMARY_PATH)
    print("E2 transition figure saved:", save_figure(build_transition_heatmap(summary), "e2_faithfulness_transition"))


if __name__ == "__main__":
    main()
