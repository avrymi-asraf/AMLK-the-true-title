"""Generate the final human-versus-judge rubric-agreement figure.

Reads `results/summaries/human_validation.json` and renders within ±1 ordinal point and exact-match
agreement between three pooled annotators (disjoint subsets) and the automated rubric judge.
Pairwise preference is not plotted here — paper Figure 5 (E3 win rate) is the sole pairwise
figure. Local, CPU-only — Plotly + kaleido.

Run:
    python -m pipeline.stage_03_reference_experiments.human_validation.make_figure
"""

from __future__ import annotations

import plotly.graph_objects as go

from pipeline.common.json_io import load_json
from pipeline.common.paths import SUMMARIES_DIR
from pipeline.common.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure
from pipeline.stage_02_evaluation_instrument.rubric_judge import DIMENSIONS

SUMMARY_PATH = SUMMARIES_DIR / "human_validation.json"
FIGURE_NAME = "human_validation"


def build_f6_human_validation(summary: dict) -> go.Figure:
    """Single-panel rubric closeness: within ±1 pt and exact match per dimension."""
    rubric = summary.get("rubric", {})
    detail = rubric.get("judge_human_pooled_detail", {})

    n_annotators = rubric.get("n_annotators", 3)
    n_articles = rubric.get("n_pooled_articles", 152)

    dim_labels = [d.replace("_", " ").title() for d in DIMENSIONS]
    within_one = [detail.get(d, {}).get("within_one_pct") for d in DIMENSIONS]
    exact_match = [detail.get(d, {}).get("exact_match_pct") for d in DIMENSIONS]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Within ±1 point",
            x=dim_labels,
            y=within_one,
            marker_color=CATEGORICAL_PALETTE[0],
            text=[f"{v:.0f}%" if v is not None else "" for v in within_one],
            textposition="outside",
            customdata=[
                [
                    detail.get(dim, {}).get("human_mean"),
                    detail.get(dim, {}).get("judge_mean"),
                ]
                for dim in DIMENSIONS
            ],
            hovertemplate=(
                "%{x}<br>within ±1: %{y:.0f}%<br>"
                "human mean %{customdata[0]:.2f} · judge mean %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Exact match",
            x=dim_labels,
            y=exact_match,
            marker_color=CATEGORICAL_PALETTE[2],
            text=[f"{v:.0f}%" if v is not None else "" for v in exact_match],
            textposition="outside",
            hovertemplate="%{x}<br>exact: %{y:.0f}%<extra></extra>",
        )
    )

    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.14)
    fig.update_yaxes(range=[0, 105])

    apply_house_style(
        fig,
        title="Human validation vs. automated rubric judge",
        subtitle=f"{n_annotators} annotators · {n_articles} rubric articles · disjoint split",
        xaxis_title="Rubric dimension",
        yaxis_title="% agreement with judge",
        source_note="Pooled human scores on disjoint article subsets",
        height=520,
    )
    return fig


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"Missing {SUMMARY_PATH}. Run "
            "`python -m pipeline.stage_03_reference_experiments.human_validation.summarize`."
        )
    summary = load_json(SUMMARY_PATH)
    fig = build_f6_human_validation(summary)
    paths = save_figure(fig, FIGURE_NAME)
    print("Wrote:", ", ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    main()
