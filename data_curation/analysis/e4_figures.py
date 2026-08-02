"""F8 from `docs/obsidian/Paper Figures.md` -- the E4 model-output comparison reported in
`paper/main.tex`. It follows `scripts.e4_score`: paired rubric rows from the uncleaned-trained and
curated-trained arms become four ordinal score-distribution panels on the same axis used for E1.
This is a local, CPU-only reporting step (Plotly + kaleido); it never loads a model or calls an API.

Run:
    python -m data_curation.analysis.e4_figures \
        --rubric-jsonl outputs/results/e4/e4-rubric-scores.jsonl

Output:
    outputs/figures/f8a_e4_rubric.{html,svg,png}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_curation.analysis.plotting import apply_house_style, ordinal_palette, save_figure

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "results"
E4_RUBRIC_PATH = RESULTS_DIR / "e4" / "e4-rubric-scores.jsonl"

DIMENSIONS = ("faithfulness", "single_focus", "informativeness", "cleanliness")
DIMENSION_LABELS = {
    "faithfulness": "Faithfulness",
    "single_focus": "Single-focus",
    "informativeness": "Informativeness",
    "cleanliness": "Cleanliness",
}
ARM_KEYS = ("raw_scores", "curated_scores")
ARM_LABELS = ("A: uncleaned", "B: curated")


def load_rubric_rows(path: Path, *, expected_n: int | None = None) -> list[dict]:
    """Load and validate paired four-dimension rubric scores from `scripts.e4_score`."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for arm_key in ARM_KEYS:
                scores = row.get(arm_key)
                if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
                    raise ValueError(
                        f"line {line_number}: {arm_key} must contain all four rubric dimensions"
                    )
                if any(
                    isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5
                    for score in scores.values()
                ):
                    raise ValueError(
                        f"line {line_number}: scores must be integers from 1 to 5"
                    )
            rows.append(row)
    if expected_n is not None and len(rows) != expected_n:
        raise ValueError(f"expected {expected_n} rubric rows, found {len(rows)}")
    return rows


def summarize_rubric_rows(rows: list[dict]) -> dict:
    """Aggregate score counts and means for each dimension and training arm."""
    if not rows:
        raise ValueError("rubric rows are empty")
    by_dimension: dict[str, dict] = {}
    for dimension in DIMENSIONS:
        raw = [row["raw_scores"][dimension] for row in rows]
        curated = [row["curated_scores"][dimension] for row in rows]
        raw_counts = Counter(raw)
        curated_counts = Counter(curated)
        by_dimension[dimension] = {
            "raw_counts": [raw_counts[level] for level in range(1, 6)],
            "curated_counts": [curated_counts[level] for level in range(1, 6)],
            "raw_mean": sum(raw) / len(raw),
            "curated_mean": sum(curated) / len(curated),
        }
    return {"n_pairs": len(rows), "by_dimension": by_dimension}


def build_f8a_rubric_distributions(summary: dict) -> go.Figure:
    """F8 -- ordinal score distributions for both E4 training arms on all four dimensions."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[DIMENSION_LABELS[dimension] for dimension in DIMENSIONS],
        horizontal_spacing=0.17,
        vertical_spacing=0.22,
    )
    colors = ordinal_palette(5)
    n_pairs = summary["n_pairs"]

    for index, dimension in enumerate(DIMENSIONS):
        row = index // 2 + 1
        col = index % 2 + 1
        dimension_summary = summary["by_dimension"][dimension]
        for level, color in zip(range(1, 6), colors):
            percentages = [
                100 * dimension_summary[f"{arm}_counts"][level - 1] / n_pairs
                for arm in ("raw", "curated")
            ]
            fig.add_trace(
                go.Bar(
                    y=list(ARM_LABELS),
                    x=percentages,
                    name=f"Score {level}",
                    legendgroup=f"score-{level}",
                    showlegend=index == 0,
                    orientation="h",
                    marker_color=color,
                    text=[f"{value:.0f}%" if value >= 8 else "" for value in percentages],
                    textposition="inside",
                    hovertemplate=(
                        f"{DIMENSION_LABELS[dimension]}<br>"
                        "%{y}<br>"
                        f"Score {level}: %{{x:.1f}}%<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(range=[0, 100], ticksuffix="%", row=row, col=col)
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=list(reversed(ARM_LABELS)),
            row=row,
            col=col,
        )

    fig.update_layout(barmode="stack")
    apply_house_style(
        fig,
        "E4 rubric distributions by training arm",
        subtitle=(
            f"Share at each score from 1 (worst) to 5 (best), "
            f"shared seeded n={n_pairs} test subset"
        ),
        width=1200,
        height=620,
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            x=0.5,
            xanchor="center",
            traceorder="normal",
        ),
        margin=dict(l=170, r=50, t=105, b=110),
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rubric-jsonl",
        type=Path,
        default=E4_RUBRIC_PATH,
        help="Paired E4 rubric scores written by scripts.e4_score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rubric_rows(args.rubric_jsonl, expected_n=120)
    summary = summarize_rubric_rows(rows)
    figure = build_f8a_rubric_distributions(summary)
    print("F8 saved:", save_figure(figure, "f8a_e4_rubric"))


if __name__ == "__main__":
    main()
