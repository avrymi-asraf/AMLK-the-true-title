"""An interim, explicitly-labeled appendix figure for `paper/main.tex` (Appendix: Interim
fine-tuned vs.\\ zero-shot comparison): the fine-tuned Arm B checkpoint against zero-shot
\\texttt{dicta-il/dictalm2.0-instruct} on the frozen test split, broken out by headline edit
sub-type and by stratum. This is *not* the paper's E4 design (Arm A vs.\\ Arm B, isolating the
headline-repair effect) -- Arm A does not exist yet -- so it cannot resolve E4; it is reported
only as an early caution sign while the real comparison is still in progress. Reads
`outputs/results/finetuned-by-edit-type.json`, a result artifact produced by an external
evaluation run (LLM-judge faithfulness/fluency + ROUGE-1/BERTScore on 580 matched test rows).
Local, CPU-only: Plotly + kaleido for rendering, no GPU/API.

Run:
    python -m data_curation.analysis.finetuned_baseline_figures

Output:
    outputs/figures/sx17_finetuned_vs_zeroshot.{html,svg,png}
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_curation.analysis.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure
from data_curation.utils.json_io import load_json

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "results"
FINETUNED_BY_EDIT_TYPE_PATH = RESULTS_DIR / "finetuned-by-edit-type.json"

EDIT_TYPE_ORDER = ["kept", "full_rewrite", "light_edit", "boilerplate_stripped", "pipes_removed"]
SMALL_N_THRESHOLD = 10  # groups below this are annotated as indicative only, not tested


def build_sx17_finetuned_vs_zeroshot(data: dict) -> go.Figure:
    """Grouped bars of fine-tuned vs.\\ zero-shot-base LLM-judge faithfulness and fluency, by
    headline edit sub-type. Every sub-type and both strata favor the zero-shot base model on both
    judge dimensions -- the opposite of what training on curated headlines is meant to show.
    """
    finetuned = data["finetuned_by_edit_type"]
    base = data["base_by_edit_type"]
    order = [e for e in EDIT_TYPE_ORDER if e in finetuned]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["LLM-judge faithfulness", "LLM-judge fluency"],
                         horizontal_spacing=0.1)

    def x_labels() -> list[str]:
        labels = []
        for edit_type in order:
            n = finetuned[edit_type]["n"]
            suffix = "*" if n < SMALL_N_THRESHOLD else ""
            labels.append(f"{edit_type}{suffix}<br>(n={n})")
        return labels

    for col, (metric, title) in enumerate([("faith_mean", "faithfulness"), ("flu_mean", "fluency")], start=1):
        fig.add_trace(go.Bar(
            x=x_labels(), y=[finetuned[e][metric] for e in order], name="Fine-tuned (Arm B)",
            marker_color=CATEGORICAL_PALETTE[0], showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Bar(
            x=x_labels(), y=[base[e][metric] for e in order], name="Zero-shot base",
            marker_color=CATEGORICAL_PALETTE[1], showlegend=(col == 1),
        ), row=1, col=col)

    fig.update_layout(barmode="group")
    apply_house_style(
        fig,
        "Interim: fine-tuned Arm B vs. zero-shot base, by headline edit sub-type",
        subtitle=(
            f"n={data['n_matched']} matched test rows -- NOT the Arm A vs. Arm B curation test (Arm A "
            "does not exist yet); reported only as an early caution sign. *n<10, indicative only."
        ),
        yaxis_title="mean judge score (1-5)",
        source_note="Source: outputs/results/finetuned-by-edit-type.json (external evaluation run)",
        width=1200,
        height=480,
    )
    fig.update_yaxes(range=[0, 5.5])
    fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    return fig


def main() -> None:
    data = load_json(FINETUNED_BY_EDIT_TYPE_PATH)
    fig = build_sx17_finetuned_vs_zeroshot(data)
    print("sx17 saved:", save_figure(fig, "sx17_finetuned_vs_zeroshot"))


if __name__ == "__main__":
    main()
