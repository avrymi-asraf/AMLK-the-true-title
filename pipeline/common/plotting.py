"""Shared visual style for the final paper figures."""

from __future__ import annotations

import plotly.colors
import plotly.graph_objects as go
import plotly.io as pio

from pipeline.common.paths import FIGURES_DIR


FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"
INK = "#1a1a1a"        # body text / axis titles
MUTED = "#6b6b6b"      # subtitle / source note / gridlines' text
GRIDLINE = "#e6e6e6"
AXIS_LINE = "#b0b0b0"

# Okabe & Ito (2008) 8-color palette — colorblind-safe categorical distinctions (defect types,
# strata, systems). Never use a "rainbow" scale for anything ordered.
CATEGORICAL_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def ordinal_palette(n: int = 5) -> list[str]:
    """A perceptually-ordered, colorblind-safe sequential palette for ordinal data (e.g. the
    rubric's 1-to-5 scores). Single-hue Blues, light-to-dark, so rank order is visible even in
    grayscale print. Clipped away from pure white/near-black so every step stays legible against
    a white figure background.
    """
    stops = [0.15 + 0.80 * i / max(n - 1, 1) for i in range(n)]
    return plotly.colors.sample_colorscale("Blues", stops)


def _build_template() -> go.layout.Template:
    """Construct the shared "amlk" Plotly template: one place that sets fonts, gridlines, and
    colorway for every figure so the paper doesn't look like a patchwork of default themes.
    """
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_FAMILY, size=14, color=INK),
            title=dict(font=dict(size=20, color=INK), x=0.02, xanchor="left", y=0.96, yanchor="top"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            colorway=CATEGORICAL_PALETTE,
            margin=dict(l=80, r=60, t=100, b=90),
            xaxis=dict(
                showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False,
                showline=True, linecolor=AXIS_LINE, ticks="outside", tickcolor=AXIS_LINE,
                title=dict(font=dict(size=14, color=MUTED)),
            ),
            yaxis=dict(
                showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False,
                showline=False, ticks="outside", tickcolor=AXIS_LINE,
                title=dict(font=dict(size=14, color=MUTED)),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
            hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
        )
    )


pio.templates["amlk"] = _build_template()


def apply_house_style(
    fig: go.Figure,
    title: str,
    *,
    subtitle: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    source_note: str | None = None,
    width: int = 1100,
    height: int = 650,
) -> go.Figure:
    """Apply the shared editorial layout: bold left-aligned title, optional muted subtitle
    directly beneath it, optional small source note pinned bottom-right. Every figure in
    `figures.py` should route through this instead of setting layout ad hoc.
    """
    title_spec: dict = {"text": f"<b>{title}</b>"}
    if subtitle:
        title_spec["subtitle"] = {"text": subtitle, "font": {"size": 13, "color": MUTED}}

    fig.update_layout(
        template="amlk",
        title=title_spec,
        width=width,
        height=height,
    )
    if xaxis_title is not None:
        fig.update_xaxes(title_text=xaxis_title)
    if yaxis_title is not None:
        fig.update_yaxes(title_text=yaxis_title)

    if source_note:
        fig.add_annotation(
            text=source_note,
            xref="paper", yref="paper", x=1, y=-0.15,
            xanchor="right", yanchor="top", showarrow=False,
            font=dict(size=11, color=MUTED, style="italic"),
        )

    return fig


def save_figure(fig: go.Figure, name: str, *, output_dir=FIGURES_DIR) -> dict[str, object]:
    """Write the one canonical PNG for a final figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.write_image(str(path), scale=3)
    return {"png": path}
