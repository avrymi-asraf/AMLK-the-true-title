"""F9a human spot-check figure for the paper appendix (replaces the judge-only test-retest plot as
"Figure 5"). Reads the one fully-submitted annotator's human annotations
(`data_curation/artifacts/human_annotations/amit.jsonl`, the only completed submission under the
disjoint F9a worklist split) and the
matching automated judge outputs (`e1_rubric_scores.jsonl`, `e3_pairwise.jsonl`), then reports plain
exact-match / within-1-point agreement per rubric dimension plus pairwise-winner agreement — no
kappa, so the numbers are readable without a stats background. Local, CPU-only: Plotly + kaleido.

Run:
    python -m data_curation.analysis.human_check_figure

Output:
    outputs/figures/f9a_human_agreement.{html,svg,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from data_curation.analysis.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure

ROOT = Path(__file__).resolve().parents[2]
AMIT_ANNOTATIONS_PATH = ROOT / "data_curation" / "artifacts" / "human_annotations" / "amit.jsonl"
E1_SCORES_PATH = ROOT / "outputs" / "results" / "e1_rubric_scores.jsonl"
E3_PAIRWISE_PATH = ROOT / "outputs" / "results" / "e3_pairwise.jsonl"

DIMENSIONS = ["faithfulness", "single_focus", "informativeness", "cleanliness"]
DIMENSION_LABELS = {
    "faithfulness": "Faithfulness",
    "single_focus": "Single-focus",
    "informativeness": "Informativeness",
    "cleanliness": "Cleanliness",
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_human_rubric_scores(path: Path = AMIT_ANNOTATIONS_PATH) -> dict[str, dict[str, int]]:
    return {
        rec["hesum_id"]: rec["scores"]
        for rec in _load_jsonl(path)
        if rec.get("task") == "rubric"
    }


def load_human_pairwise(path: Path = AMIT_ANNOTATIONS_PATH) -> list[dict]:
    return [rec for rec in _load_jsonl(path) if rec.get("task") == "pairwise"]


def load_judge_rubric_scores(path: Path = E1_SCORES_PATH) -> dict[str, dict[str, int]]:
    scores: dict[str, dict[str, int]] = {}
    for row in _load_jsonl(path):
        scores[row["hesum_id"]] = {
            dim: row["scores"][dim]["score"]
            for dim in DIMENSIONS
            if dim in row.get("scores", {})
        }
    return scores


def load_judge_pairwise_outcomes(path: Path = E3_PAIRWISE_PATH) -> dict[str, str]:
    return {
        row["hesum_id"]: row["outcome"]
        for row in _load_jsonl(path)
        if row.get("outcome") in {"curated", "original", "tie"}
    }


def rubric_agreement_pct(
    human: dict[str, dict[str, int]], judge: dict[str, dict[str, int]]
) -> dict[str, dict[str, float]]:
    """Plain exact-match and within-1-point agreement per dimension (no kappa)."""
    shared = sorted(set(human) & set(judge))
    result: dict[str, dict[str, float]] = {}
    for dim in DIMENSIONS:
        exact = within1 = n = 0
        for hesum_id in shared:
            if dim in human[hesum_id] and dim in judge[hesum_id]:
                a, j = human[hesum_id][dim], judge[hesum_id][dim]
                n += 1
                exact += a == j
                within1 += abs(a - j) <= 1
        result[dim] = {
            "n": n,
            "exact_pct": 100 * exact / n if n else float("nan"),
            "within1_pct": 100 * within1 / n if n else float("nan"),
        }
    return result


def pairwise_agreement_pct(human_records: list[dict], judge_outcomes: dict[str, str]) -> dict:
    """Plain percent agreement on curated/original/tie between one human annotator and the judge."""
    from evaluation.pairwise_judge import curated_wins

    human_outcome = {
        rec["hesum_id"]: curated_wins(rec["winner"], rec["slot_map"].get("a") == "curated")
        for rec in human_records
    }
    shared = sorted(set(human_outcome) & set(judge_outcomes))
    if not shared:
        return {"n": 0, "agree_pct": float("nan")}
    agree = sum(human_outcome[h] == judge_outcomes[h] for h in shared)
    return {"n": len(shared), "agree_pct": 100 * agree / len(shared)}


def build_figure(rubric: dict[str, dict[str, float]], pairwise: dict) -> go.Figure:
    fig = go.Figure()
    labels = [DIMENSION_LABELS[d] for d in DIMENSIONS]
    fig.add_trace(go.Bar(
        x=labels, y=[rubric[d]["within1_pct"] for d in DIMENSIONS],
        name="Within 1 point", marker_color=CATEGORICAL_PALETTE[5],
        text=[f"{rubric[d]['within1_pct']:.0f}%" for d in DIMENSIONS], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=[rubric[d]["exact_pct"] for d in DIMENSIONS],
        name="Exact match", marker_color=CATEGORICAL_PALETTE[0],
        text=[f"{rubric[d]['exact_pct']:.0f}%" for d in DIMENSIONS], textposition="outside",
    ))

    n_rubric = next(iter(rubric.values()))["n"]
    subtitle = (
        f"One annotator, n={n_rubric} rubric rows -- pairwise headline preference agreed "
        f"{pairwise['agree_pct']:.0f}% of the time (n={pairwise['n']})"
    )
    apply_house_style(
        fig,
        "Judge vs. human spot-check (F9a)",
        subtitle=subtitle,
        yaxis_title="share of rows",
        source_note="Source: F9a human annotations vs. judge outputs",
        width=900,
        height=480,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 115], ticksuffix="%")
    return fig


def main() -> None:
    human_rubric = load_human_rubric_scores()
    judge_rubric = load_judge_rubric_scores()
    rubric = rubric_agreement_pct(human_rubric, judge_rubric)

    human_pairwise = load_human_pairwise()
    judge_pairwise = load_judge_pairwise_outcomes()
    pairwise = pairwise_agreement_pct(human_pairwise, judge_pairwise)

    fig = build_figure(rubric, pairwise)
    paths = save_figure(fig, "f9a_human_agreement")
    print("Wrote:", ", ".join(str(p) for p in paths.values()))
    print("rubric:", json.dumps(rubric, indent=2))
    print("pairwise:", json.dumps(pairwise, indent=2))


if __name__ == "__main__":
    main()
