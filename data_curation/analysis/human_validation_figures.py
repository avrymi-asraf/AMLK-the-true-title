"""F9a appendix figure: agreement heatmap (human–human, judge–human, pilot test-retest kappa).

Reads `human_validation_summary.json` and optional `rubric_pilot.json` for the judge test-retest
reference row. Local, CPU-only — Plotly + kaleido.

Run:
    python -m data_curation.analysis.human_validation_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from data_curation.analysis.plotting import apply_house_style, save_figure
from evaluation.rubric_judge import DIMENSIONS

SUMMARY_PATH = Path(__file__).resolve().parents[2] / "outputs" / "results" / "human_validation_summary.json"
PILOT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "results" / "rubric_pilot.json"
FIGURE_NAME = "f9a_judge_validation"


def load_pilot_test_retest(path: Path = PILOT_PATH) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("test_retest_kappa", {})


def build_f9a_heatmap(summary: dict, pilot_kappa: dict[str, float]) -> go.Figure:
    """Heatmap: rows = comparison types, columns = rubric dimensions (+ pairwise)."""
    rubric = summary.get("rubric", {})
    row_labels = []
    matrix = []

    for pair_key, kappas in rubric.get("human_human_pairs", {}).items():
        label = pair_key.replace("_vs_", " vs ")
        row_labels.append(f"Human–human ({label})")
        matrix.append([kappas.get(dim) for dim in DIMENSIONS])

    if not rubric.get("human_human_pairs") and rubric.get("human_human"):
        row_labels.append("Human–human")
        matrix.append([rubric["human_human"].get(dim) for dim in DIMENSIONS])

    judge_human = rubric.get("judge_human", {})
    if judge_human:
        for aid, kappas in judge_human.items():
            row_labels.append(f"Judge vs {aid}")
            matrix.append([kappas.get(dim) for dim in DIMENSIONS])

    pooled = rubric.get("judge_human_pooled")
    if pooled and rubric.get("split_mode") == "disjoint":
        row_labels.append("Judge vs human (pooled)")
        matrix.append([pooled.get(dim) for dim in DIMENSIONS])
    else:
        annotators = rubric.get("annotator_ids", [])
        if rubric.get("judge_human_a"):
            label = f"Judge vs {annotators[0]}" if annotators else "Judge vs human A"
            row_labels.append(label)
            matrix.append([rubric["judge_human_a"].get(dim) for dim in DIMENSIONS])
        if rubric.get("judge_human_b") and len(annotators) > 1:
            row_labels.append(f"Judge vs {annotators[1]}")
            matrix.append([rubric["judge_human_b"].get(dim) for dim in DIMENSIONS])

    if pilot_kappa:
        row_labels.append("Judge test-retest (pilot)")
        matrix.append([pilot_kappa.get(dim) for dim in DIMENSIONS])

    col_labels = [d.replace("_", " ") for d in DIMENSIONS]

    z = []
    text = []
    for row in matrix:
        z_row = []
        t_row = []
        for val in row:
            if val is None:
                z_row.append(float("nan"))
                t_row.append("")
            else:
                z_row.append(val)
                t_row.append(f"{val:.3f}")
        z.append(z_row)
        text.append(t_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=col_labels,
            y=row_labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="κ"),
        )
    )

    subtitle_parts = []
    completion = summary.get("completion", {})
    if completion.get("expected_tasks"):
        subtitle_parts.append(
            f"{completion.get('submitted_tasks', 0)}/{completion['expected_tasks']} tasks submitted"
        )
    pairwise = summary.get("pairwise", {})
    if pairwise:
        first = next(iter(pairwise.values()), {})
        if first.get("quadratic_kappa") is not None:
            subtitle_parts.append(f"pairwise κ={first['quadratic_kappa']:.3f}")

    apply_house_style(
        fig,
        title="F9a — Judge validation agreement",
        subtitle=" · ".join(subtitle_parts) if subtitle_parts else None,
        xaxis_title="Rubric dimension",
        yaxis_title="Comparison",
        source_note="Quadratically weighted Cohen's κ · human validation round",
        height=450 + 40 * len(row_labels),
    )
    return fig


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"Missing {SUMMARY_PATH}. Run human_validation_results.py after annotations are collected."
        )
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    pilot_kappa = load_pilot_test_retest()
    fig = build_f9a_heatmap(summary, pilot_kappa)
    paths = save_figure(fig, FIGURE_NAME)
    print("Wrote:", ", ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    main()
