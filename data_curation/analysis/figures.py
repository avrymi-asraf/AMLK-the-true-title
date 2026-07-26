"""F1 and F2 from `docs/obsidian/Paper Figures.md` — the two dataset-review figures buildable
straight from artifacts already on disk, no rubric-judge budget needed. Both read the row-label
artifact (`row_labels.py`) exclusively, so the pipeline's real counts and the figures can never
drift apart. Local, CPU-only: Plotly + kaleido for rendering, no GPU/API.

Run:
    python -m data_curation.analysis.figures

Output:
    outputs/figures/f1_curation_funnel.{html,svg,png}
    outputs/figures/f2_defect_prevalence.{html,svg,png}
"""

from __future__ import annotations

import plotly.graph_objects as go

from data_curation.analysis.plotting import CATEGORICAL_PALETTE, apply_house_style, save_figure
from data_curation.analysis.row_labels import load_row_labels


SOURCE_NOTE = "Source: data_curation pipeline · n = 10,000 raw HeSum rows"

# Semantic color roles, drawn from the shared categorical palette: blue for rows still moving
# through the pipeline, green for the clean reference group, warm hues for each distinct defect,
# gray for the three sub-labels too small to test (prevalence-only, see section 3 of the spec).
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, YELLOW, BLACK = CATEGORICAL_PALETTE
GRAY = "#999999"


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a `#rrggbb` color to an `rgba(...)` string, for translucent Sankey links."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def compute_funnel_counts(rows: list[dict]) -> dict[str, int]:
    """Partition the 10,000 rows into the disjoint funnel stages F1 draws.

    The two deterministic filters are independent, so a row can fail one, the other, both, or
    neither — this recovers all four outcomes from the row-label artifact rather than assuming
    the filters are applied in sequence.
    """
    over_budget_only = sum(r["over_token_budget"] and not r["multi_pipe"] for r in rows)
    multi_pipe_only = sum(r["multi_pipe"] and not r["over_token_budget"] for r in rows)
    both_filters = sum(r["over_token_budget"] and r["multi_pipe"] for r in rows)
    reached = sum(r["reached_model_curation"] for r in rows)
    usable = sum(r["source_label"] == "usable" for r in rows)
    kept = sum(r["headline_action"] == "kept" for r in rows)
    rewritten = sum(r["headline_action"] == "rewritten" for r in rows)

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


def build_f1_curation_funnel(rows: list[dict]) -> go.Figure:
    """F1 — Sankey funnel: raw HeSum through both deterministic filters (shown as separate,
    converging flows, not one combined "filtering" node) to the usable/unusable split and the
    kept-versus-rewritten headline fork.
    """
    counts = compute_funnel_counts(rows)

    labels = [
        f"Raw HeSum ({counts['total']:,})",                                    # 0
        f"Over-budget only ({counts['over_budget_only']:,})",                  # 1
        f"Multi-pipe only ({counts['multi_pipe_only']:,})",                    # 2
        f"Both filters ({counts['both_filters']:,})",                         # 3
        f"Reached model curation ({counts['reached_model_curation']:,})",      # 4
        f"Usable ({counts['usable']:,})",                                       # 5
        f"Unusable ({counts['unusable']:,})",                                   # 6
        f"Headline kept ({counts['headline_kept']:,})",                        # 7
        f"Headline rewritten ({counts['headline_rewritten']:,})",              # 8
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

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            color=node_colors,
            pad=22,
            thickness=20,
            line=dict(color="white", width=1),
        ),
        link=dict(
            source=[s for s, _, _ in links],
            target=[t for _, t, _ in links],
            value=[v for _, _, v in links],
            color=[_rgba(node_colors[s], 0.35) for s, _, _ in links],
        ),
        textfont=dict(size=13, color="#1a1a1a", family="Helvetica Neue, Helvetica, Arial, sans-serif"),
    ))

    apply_house_style(
        fig,
        "What the curation pipeline did",
        subtitle="10,000 raw HeSum rows, through both deterministic filters, to the final usable/rewritten split",
        source_note=SOURCE_NOTE,
        width=1150,
        height=620,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=110, b=90))
    return fig


def compute_stratum_counts(rows: list[dict]) -> list[tuple[str, int, str]]:
    """Defect-stratum sizes (label, count, color role), matching the analysis-strata table in
    the design spec exactly — `S0 clean` through `S5`'s three constituent labels broken out.
    """
    clean = sum(
        r["reached_model_curation"] and r["source_label"] == "usable"
        and r["headline_action"] == "kept"
        for r in rows
    )
    multi_pipe = sum(r["multi_pipe"] for r in rows)
    multiple_items = sum(r["source_label"] == "unusable_multiple_independent_items" for r in rows)
    rewritten = sum(r["headline_action"] == "rewritten" for r in rows)
    insufficient = sum(r["source_label"] == "unusable_insufficient_substantive_content" for r in rows)
    not_in_text = sum(r["source_label"] == "unusable_substantive_content_not_in_text" for r in rows)
    damaged = sum(r["source_label"] == "unusable_damaged_or_fragmentary_text" for r in rows)

    return [
        ("clean (S0)", clean, GREEN),
        ("headline_rewritten (S4)", rewritten, ORANGE),
        ("multi_pipe_headline (S2)", multi_pipe, VERMILLION),
        ("multiple_independent_items (S3)", multiple_items, PURPLE),
        ("insufficient_substantive_content (S5)", insufficient, GRAY),
        ("substantive_content_not_in_text (S5)", not_in_text, GRAY),
        ("damaged_or_fragmentary_text (S5)", damaged, GRAY),
    ]


def build_f2_defect_prevalence(rows: list[dict]) -> go.Figure:
    """F2 — horizontal bars of stratum size with corpus-share annotations, including the three
    small `other_unusable` sub-labels (gray) so the long tail stays visible even though the
    statistical analysis collapses and excludes them.
    """
    total = len(rows)
    strata = compute_stratum_counts(rows)
    strata_sorted = sorted(strata, key=lambda item: item[1])  # ascending, so the biggest bar is on top

    fig = go.Figure(go.Bar(
        x=[count for _, count, _ in strata_sorted],
        y=[label for label, _, _ in strata_sorted],
        orientation="h",
        marker_color=[color for _, _, color in strata_sorted],
        text=[f"{count:,}  ({count / total:.1%})" for _, count, _ in strata_sorted],
        textposition="outside",
        cliponaxis=False,
    ))

    apply_house_style(
        fig,
        "How much of HeSum each defect touches",
        subtitle="Share of the full 10,000-row corpus; strata overlap by design and gray bars are prevalence-only (n too small to test)",
        xaxis_title="rows",
        source_note=SOURCE_NOTE,
        width=1250,
        height=560,
    )
    fig.update_layout(margin=dict(l=340))
    fig.update_xaxes(range=[0, max(count for _, count, _ in strata) * 1.18])
    return fig


def main() -> None:
    """Build and save F1 and F2."""
    rows = load_row_labels()

    f1 = build_f1_curation_funnel(rows)
    f1_paths = save_figure(f1, "f1_curation_funnel")
    print(f"F1 saved: {f1_paths}")

    f2 = build_f2_defect_prevalence(rows)
    f2_paths = save_figure(f2, "f2_defect_prevalence")
    print(f"F2 saved: {f2_paths}")


if __name__ == "__main__":
    main()
