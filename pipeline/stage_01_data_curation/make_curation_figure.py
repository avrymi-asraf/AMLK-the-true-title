"""Generate the final curation-funnel figure from the frozen row ledger."""

from __future__ import annotations

import plotly.graph_objects as go

from pipeline.common.json_io import load_json
from pipeline.common.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure
from pipeline.common.paths import DATA_CURATION_ARTIFACTS_DIR


ROW_LABELS_PATH = DATA_CURATION_ARTIFACTS_DIR / "row_labels.json"
SOURCE_NOTE = "Source: data-curation pipeline · n = 10,000 raw HeSum rows"
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, _SKY, _YELLOW, _BLACK = CATEGORICAL_PALETTE
GRAY = "#999999"


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def compute_funnel_counts(rows: list[dict]) -> dict[str, int]:
    """Partition rows into the disjoint flows shown in the final funnel."""
    over_budget_only = sum(row["over_token_budget"] and not row["multi_pipe"] for row in rows)
    multi_pipe_only = sum(row["multi_pipe"] and not row["over_token_budget"] for row in rows)
    both_filters = sum(row["over_token_budget"] and row["multi_pipe"] for row in rows)
    reached = sum(row["reached_model_curation"] for row in rows)
    usable = sum(row["source_label"] == "usable" for row in rows)
    kept = sum(row["headline_action"] == "kept" for row in rows)
    rewritten = sum(row["headline_action"] == "rewritten" for row in rows)
    return {
        "total": len(rows),
        "over_budget_only": over_budget_only,
        "multi_pipe_only": multi_pipe_only,
        "both_filters": both_filters,
        "reached_model_curation": reached,
        "usable": usable,
        "unusable": reached - usable,
        "headline_kept": kept,
        "headline_rewritten": rewritten,
    }


def build_curation_funnel(rows: list[dict]) -> go.Figure:
    counts = compute_funnel_counts(rows)
    labels = [
        f"Raw HeSum ({counts['total']:,})",
        f"Over-budget only ({counts['over_budget_only']:,})",
        f"Multi-pipe only ({counts['multi_pipe_only']:,})",
        f"Both filters ({counts['both_filters']:,})",
        f"Reached model curation ({counts['reached_model_curation']:,})",
        f"Usable ({counts['usable']:,})",
        f"Unusable ({counts['unusable']:,})",
        f"Headline kept ({counts['headline_kept']:,})",
        f"Headline rewritten ({counts['headline_rewritten']:,})",
    ]
    node_colors = [GRAY, ORANGE, VERMILLION, PURPLE, BLUE, GREEN, VERMILLION, BLUE, ORANGE]
    links = [
        (0, 1, counts["over_budget_only"]),
        (0, 2, counts["multi_pipe_only"]),
        (0, 3, counts["both_filters"]),
        (0, 4, counts["reached_model_curation"]),
        (4, 5, counts["usable"]),
        (4, 6, counts["unusable"]),
        (5, 7, counts["headline_kept"]),
        (5, 8, counts["headline_rewritten"]),
    ]
    figure = go.Figure(go.Sankey(
        arrangement="snap",
        node={
            "label": labels,
            "color": node_colors,
            "pad": 22,
            "thickness": 20,
            "line": {"color": "white", "width": 1},
        },
        link={
            "source": [source for source, _, _ in links],
            "target": [target for _, target, _ in links],
            "value": [value for _, _, value in links],
            "color": [_rgba(node_colors[source], 0.35) for source, _, _ in links],
        },
        textfont={"size": 13, "color": "#1a1a1a", "family": "Helvetica Neue, Helvetica, Arial, sans-serif"},
    ))
    apply_house_style(
        figure,
        "What the curation pipeline did",
        subtitle="10,000 raw HeSum rows through deterministic filters and model-assisted curation",
        source_note=SOURCE_NOTE,
        width=1150,
        height=620,
    )
    figure.update_layout(margin={"l": 20, "r": 20, "t": 110, "b": 90})
    return figure


def main() -> None:
    rows = load_json(ROW_LABELS_PATH)
    print("Curation funnel saved:", save_figure(build_curation_funnel(rows), "curation_funnel"))


if __name__ == "__main__":
    main()
