"""F9 from `docs/obsidian/Paper Figures.md` — the zero-shot DictaLM2.0-vs-reference figure reported
in `paper/main.tex` (Section: Results, "Zero-shot model output vs. reference on the same rubric
axis"). Reads the precomputed baseline-vs-reference comparison
(`outputs/results/baseline-rubric-comparison.json`), produced independently of the E4 fine-tuning
run this figure does not depend on. Local, CPU-only: Plotly + kaleido for rendering, no GPU/API.

Run:
    python -m data_curation.analysis.baseline_reliability_figures

Output:
    outputs/figures/f9_baseline_vs_reference_rubric.{html,svg,png}
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_curation.analysis.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure
from data_curation.utils.json_io import load_json

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "results"
BASELINE_COMPARISON_PATH = RESULTS_DIR / "baseline-rubric-comparison.json"

DIMENSIONS = ["faithfulness", "single_focus", "informativeness", "cleanliness"]
DIMENSION_LABELS = {
    "faithfulness": "Faithfulness",
    "single_focus": "Single-focus",
    "informativeness": "Informativeness",
    "cleanliness": "Cleanliness",
}
STRATA_ORDER = ["S0_clean", "S2_multi_pipe_headline", "S3_multiple_independent_items", "S4_headline_rewritten"]
STRATUM_LABELS = {
    "S0_clean": "S0 clean",
    "S2_multi_pipe_headline": "S2 multi-pipe",
    "S3_multiple_independent_items": "S3 multi-item",
    "S4_headline_rewritten": "S4 rewritten",
}


def build_f9_baseline_vs_reference(comparison: dict) -> go.Figure:
    """F9 — grouped bars of reference vs.\\ zero-shot-model median score, one panel per stratum.

    The clearest signal is the S2 single-focus *reversal*: the zero-shot model scores higher than
    the defective reference on exactly the dimension the multi-pipe defect damages, direct
    evidence that a defective reference can lose to a generic summarizer (main.tex, Section:
    Zero-shot model output vs. reference on the same rubric axis).
    """
    fig = make_subplots(rows=1, cols=4, subplot_titles=[STRATUM_LABELS[s] for s in STRATA_ORDER],
                         shared_yaxes=True, horizontal_spacing=0.05)

    for col, stratum in enumerate(STRATA_ORDER, start=1):
        by_dim = comparison["by_stratum"][stratum]
        fig.add_trace(go.Bar(
            x=[DIMENSION_LABELS[d] for d in DIMENSIONS],
            y=[by_dim[d]["reference_median"] for d in DIMENSIONS],
            name="Reference", marker_color=CATEGORICAL_PALETTE[1], showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Bar(
            x=[DIMENSIONS[d] if False else DIMENSION_LABELS[d] for d in DIMENSIONS],
            y=[by_dim[d]["model_median"] for d in DIMENSIONS],
            name="Zero-shot model", marker_color=CATEGORICAL_PALETTE[0], showlegend=(col == 1),
        ), row=1, col=col)

    fig.update_layout(barmode="group")
    apply_house_style(
        fig,
        "Zero-shot DictaLM2.0 vs. reference headline, same rubric axis",
        subtitle=(
            f"n={comparison['n_paired']} paired predictions, {comparison['model']} -- "
            "S2 single-focus is the reversal: the model beats a defective reference on the dimension it damages"
        ),
        yaxis_title="median rubric score (1-5)",
        source_note="Source: outputs/results/baseline-rubric-comparison.json",
        width=1300,
        height=500,
    )
    fig.update_yaxes(range=[0, 5.5])
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
    return fig


def main() -> None:
    comparison = load_json(BASELINE_COMPARISON_PATH)
    fig = build_f9_baseline_vs_reference(comparison)
    print("F9 saved:", save_figure(fig, "f9_baseline_vs_reference_rubric"))


if __name__ == "__main__":
    main()
