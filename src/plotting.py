"""Shared dark-theme styling for matplotlib figures.

Keeps every generated PNG consistent with the Streamlit dashboard's dark
"liquid glass" look: deep near-black background, light ink, and an F1-red
accent. Call :func:`apply_dark_theme` once before plotting, and
:func:`style_fig` on a figure just before saving (so SHAP and other plots that
create their own figure also get the dark background).
"""
from __future__ import annotations

import matplotlib as mpl

# Palette (shared with the dashboard CSS).
BG = "#0E1117"          # app background
SURFACE = "#161A23"     # panel background
INK = "#ECEDEE"         # primary text
MUTED = "#9BA1AC"       # secondary text / ticks
GRID = "#262B36"        # gridlines / borders
ACCENT = "#E10600"      # F1 red
ACCENT_SOFT = "#FF4D5E"  # lighter red for highlights

# Sequential reds for bar charts (low -> high).
RED_SEQUENCE = ["#5A0A0A", "#8C0F12", "#C01016", "#E10600", "#FF4D5E"]


def apply_dark_theme() -> None:
    """Set global matplotlib rcParams for the dark dashboard theme."""
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "legend.frameon": False,
            "legend.labelcolor": INK,
        }
    )


def style_fig(fig) -> None:
    """Force a figure and its axes onto the dark background (post-hoc).

    Recolors backgrounds, spines, tick labels, axis labels, titles, and free
    text annotations. Needed for third-party plots (e.g. SHAP) that draw with
    hard-coded black text and white axes, ignoring rcParams.
    """
    fig.patch.set_facecolor(BG)
    for ax in fig.get_axes():
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=INK, which="both")
        for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            lbl.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        if ax.get_title():
            ax.title.set_color(INK)
        # Free-text annotations (e.g. SHAP colorbar "High"/"Low").
        for txt in ax.texts:
            txt.set_color(INK)
