"""
Shared palette, chart builders, and cached data loaders for app.py.

Colors follow the project's validated categorical order (fixed hue per
entity so a filter never repaints the survivors), a single-hue sequential
ramp for magnitude, and the reserved status colors for good/warning/
critical states -- never colors picked ad hoc per chart.
"""
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- categorical palette (fixed order, never cycled) ---------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MUTED = "#898781"

# --- status palette (reserved, never reused as a series color) -----------
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

# --- chrome / ink ----------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# fixed entity -> color maps so identity never shifts when a chart is filtered
CLUSTER_COLORS = {
    "High-Performing": BLUE,
    "Congested High-Volume": ORANGE,
    "Consistently Slow": AQUA,
    "Low-Volume / Niche": MUTED,
}
SHIP_MODE_COLORS = {
    "Same Day": BLUE,
    "First Class": ORANGE,
    "Second Class": AQUA,
    "Standard Class": YELLOW,
}
FACTORY_COLORS = {
    "Chocolate": BLUE,
    "Other": ORANGE,
    "Sugar": AQUA,
}
MODEL_COLORS = {
    "LinearRegression": BLUE,
    "DecisionTree": ORANGE,
    "RandomForest": AQUA,
    "GradientBoosting": YELLOW,
}

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
)


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT)
    fig.update_xaxes(showgrid=False, linecolor=GRIDLINE, tickfont=dict(color=INK_SECONDARY))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, tickfont=dict(color=INK_SECONDARY))
    return fig


def status_color(value: float, good: float, warning: float) -> str:
    """Higher-is-better status color: >=good -> good, >=warning -> warning, else critical."""
    if value >= good:
        return STATUS_GOOD
    if value >= warning:
        return STATUS_WARNING
    return STATUS_CRITICAL


def kpi_tile(label: str, value: str, sublabel: str, color: str):
    st.markdown(
        f"""
        <div style="border:1px solid {GRIDLINE}; border-radius:10px; padding:16px 18px;">
          <div style="color:{INK_SECONDARY}; font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.02em;">{label}</div>
          <div style="color:{color}; font-size:2rem; font-weight:700; line-height:1.2; margin-top:4px;">{value}</div>
          <div style="color:{INK_MUTED}; font-size:0.8rem; margin-top:4px;">{sublabel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def delta_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str) -> go.Figure:
    """Bar chart where fill color is a status cue (good=improvement, critical=regression)."""
    colors = [STATUS_GOOD if v <= 0 else STATUS_CRITICAL for v in df[y_col]]
    fig = go.Figure(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=colors,
            text=[f"{v:+.1f}" for v in df[y_col]],
            textposition="outside",
        )
    )
    fig.update_layout(title=title, yaxis_title=y_label)
    return style_fig(fig)


def categorical_bar(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, color_map: dict, title: str, y_label: str) -> go.Figure:
    fig = px.bar(df, x=x_col, y=y_col, color=color_col, color_discrete_map=color_map, title=title)
    fig.update_layout(yaxis_title=y_label, xaxis_title="", legend_title_text="")
    return style_fig(fig)


def categorical_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, color_map: dict, title: str, hover_data=None, size_col=None) -> go.Figure:
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, color_discrete_map=color_map, title=title,
        hover_data=hover_data, size=size_col,
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="#fcfcfb")))
    fig.update_layout(legend_title_text="")
    return style_fig(fig)


# --- cached data loaders ----------------------------------------------------
def _p(outdir: str, name: str) -> str:
    return os.path.join(outdir, name)


@st.cache_data(show_spinner=False)
def load_csv(outdir: str, name: str) -> pd.DataFrame:
    path = _p(outdir, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(outdir: str, name: str):
    path = _p(outdir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def outputs_exist(outdir: str) -> bool:
    return os.path.exists(_p(outdir, "processed.csv"))
