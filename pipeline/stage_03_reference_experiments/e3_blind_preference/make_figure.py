"""Generate the final E3 blind-preference figure from its frozen summary."""

from __future__ import annotations

import plotly.graph_objects as go

from pipeline.common.json_io import load_json
from pipeline.common.paths import REFERENCE_EXPERIMENT_ARTIFACTS_DIR
from pipeline.common.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure


SUMMARY_PATH = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "e3_pairwise_summary.json"


def build_blind_preference(summary: dict) -> go.Figure:
    rewritten = summary["rewritten"]
    required = {"n", "curated_pct", "tie_pct", "original_pct"}
    if not required.issubset(rewritten):
        raise ValueError(f"E3 rewritten summary lacks fields: {sorted(required - set(rewritten))}")
    label = f"Rewritten (n={rewritten['n']})"
    figure = go.Figure()
    for key, name, color in (
        ("curated_pct", "Curated wins", CATEGORICAL_PALETTE[0]),
        ("tie_pct", "Tie", "#cccccc"),
        ("original_pct", "Original wins", CATEGORICAL_PALETTE[3]),
    ):
        value = rewritten[key]
        figure.add_trace(go.Bar(
            y=[label],
            x=[value],
            name=name,
            orientation="h",
            marker_color=color,
            text=[f"{value:.1f}%"],
            textposition="inside",
        ))
    figure.update_layout(barmode="stack")
    apply_house_style(
        figure,
        "Blind pairwise preference: curated vs. original headline (E3)",
        subtitle="The judge compared headlines without provenance labels",
        xaxis_title="share of judgments",
        source_note="Source: frozen E3 blind pairwise summary",
        width=1000,
        height=320,
    )
    figure.update_xaxes(range=[0, 100], ticksuffix="%")
    figure.update_layout(margin={"l": 260, "b": 90})
    return figure


def main() -> None:
    summary = load_json(SUMMARY_PATH)
    print("E3 preference figure saved:", save_figure(build_blind_preference(summary), "e3_blind_preference"))


if __name__ == "__main__":
    main()
