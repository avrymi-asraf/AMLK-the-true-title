"""F8 from `docs/obsidian/Paper Figures.md` -- the E4 before/after figure reported in
`paper/main.tex` (Results, "Training on the curated corpus improves model outputs (E4)"). Reads
the aggregate summary written by `scripts/e4_score.py` (`outputs/results/e4/e4-score-summary.json`)
when available; otherwise falls back to the exact numbers already reported in the committed paper
text (n=120 seeded subset, `gemini-2.5-flash-lite`, T=0) so the figure stays reproducible without
requiring the raw judge artifact to be present on every machine. Local, CPU-only: Plotly + kaleido
for rendering, no GPU/API.

Run:
    python -m data_curation.analysis.e4_figures

Output:
    outputs/figures/f8_e4_before_after.{html,svg,png}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from data_curation.analysis.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "results"
E4_SUMMARY_PATH = RESULTS_DIR / "e4" / "e4-score-summary.json"

# Fallback: the exact numbers reported in paper/main.tex (Section: E4 results), pinned so this
# figure reproduces even when the raw e4-score-summary.json artifact isn't on disk.
FALLBACK_SUMMARY: dict[str, Any] = {
    "n_pairs": 120,
    "pointwise": {
        "by_dimension": {
            "faithfulness": {
                "raw_mean": 3.60, "curated_mean": 4.09,
                "cliffs_delta_curated_vs_raw": 0.20, "cliffs_ci": [0.07, 0.34],
            },
            "informativeness": {
                "raw_mean": 3.90, "curated_mean": 4.33,
                "cliffs_delta_curated_vs_raw": 0.19, "cliffs_ci": [0.06, 0.32],
            },
        },
    },
    "pairwise": {
        "curated_wins": 74, "raw_wins": 41, "ties": 5,
        "curated_win_rate_pct": 64.3, "wilson_ci_pct": [55.3, 72.5],
    },
}

# Only the two dimensions with a paired CI that excludes 0 (main.tex): single-focus and
# cleanliness showed no significant difference and are not plotted with fabricated bar heights.
PLOTTED_DIMENSIONS = ["faithfulness", "informativeness"]
DIMENSION_LABELS = {"faithfulness": "Faithfulness", "informativeness": "Informativeness"}


def load_e4_summary(path: Path = E4_SUMMARY_PATH) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return FALLBACK_SUMMARY


def build_f8a_rubric_bars(summary: dict) -> go.Figure:
    """F8a -- grouped bar of Arm A (uncleaned) vs. Arm B (curated) mean rubric score, for the two
    dimensions with a paired Cliff's delta CI that excludes 0 (faithfulness, informativeness).
    Single-focus and cleanliness are not plotted: main.tex reports "no significant paired
    difference" for both without per-arm means, and this figure only shows numbers already on
    the page rather than a fabricated bar height.
    """
    by_dim = summary["pointwise"]["by_dimension"]
    labels = [DIMENSION_LABELS[d] for d in PLOTTED_DIMENSIONS]
    raw_vals = [by_dim[d]["raw_mean"] for d in PLOTTED_DIMENSIONS]
    cur_vals = [by_dim[d]["curated_mean"] for d in PLOTTED_DIMENSIONS]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=raw_vals, name="Arm A (uncleaned)",
                          marker_color=CATEGORICAL_PALETTE[3],
                          text=[f"{v:.2f}" for v in raw_vals], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=cur_vals, name="Arm B (curated)",
                          marker_color=CATEGORICAL_PALETTE[0],
                          text=[f"{v:.2f}" for v in cur_vals], textposition="outside"))
    for i, d in enumerate(PLOTTED_DIMENSIONS):
        delta = by_dim[d]["cliffs_delta_curated_vs_raw"]
        lo, hi = by_dim[d]["cliffs_ci"]
        fig.add_annotation(
            x=labels[i], y=max(raw_vals[i], cur_vals[i]) + 0.7,
            text=f"\u03b4={delta:.2f} [{lo:.2f}, {hi:.2f}]",
            showarrow=False, font=dict(size=12, color="#6b6b6b"),
        )

    fig.update_layout(barmode="group")
    apply_house_style(
        fig,
        "Training on curated targets raises rubric scores (E4)",
        subtitle=(
            f"Mean score, shared seeded n={summary['n_pairs']} test subset -- single-focus and"
            " cleanliness showed no significant paired difference and are omitted"
        ),
        yaxis_title="mean rubric score (1-5)",
        source_note="Source: scripts.e4_score, outputs/results/e4/e4-score-summary.json",
        width=900,
        height=520,
    )
    fig.update_yaxes(range=[0, 5.8])
    return fig


def build_f8b_pairwise(summary: dict) -> go.Figure:
    """F8b -- blind pairwise preference between the two trained arms, drawn like F7's win-rate
    bar so the model-output comparison reads on the same visual scale as the E3 headline
    comparison.
    """
    pw = summary["pairwise"]
    n = pw["curated_wins"] + pw["raw_wins"] + pw["ties"]
    row_label = f"E4 (n={n})"
    outcomes = [
        ("curated_wins", "Curated (Arm B) wins", CATEGORICAL_PALETTE[0]),
        ("ties", "Tie", "#cccccc"),
        ("raw_wins", "Uncleaned (Arm A) wins", CATEGORICAL_PALETTE[3]),
    ]

    fig = go.Figure()
    for key, label, color in outcomes:
        pct = 100.0 * pw[key] / n
        fig.add_trace(go.Bar(
            y=[row_label], x=[pct], name=label, orientation="h", marker_color=color,
            text=[f"{pct:.1f}%"], textposition="inside",
        ))

    lo, hi = pw["wilson_ci_pct"]
    fig.update_layout(barmode="stack")
    apply_house_style(
        fig,
        "Blind pairwise preference: curated-trained vs. uncleaned-trained outputs (E4)",
        subtitle=f"Curated win rate {pw['curated_win_rate_pct']:.1f}%, Wilson 95% CI [{lo:.1f}%, {hi:.1f}%] excludes 50%",
        xaxis_title="share of judgments",
        source_note="Source: scripts.e4_score, outputs/results/e4/e4-score-summary.json",
        width=1000,
        height=300,
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_layout(margin=dict(l=260, b=90))
    return fig


def main() -> None:
    summary = load_e4_summary()
    f8a = build_f8a_rubric_bars(summary)
    print("F8a saved:", save_figure(f8a, "f8a_e4_rubric"))
    f8b = build_f8b_pairwise(summary)
    print("F8b saved:", save_figure(f8b, "f8b_e4_pairwise"))


if __name__ == "__main__":
    main()
