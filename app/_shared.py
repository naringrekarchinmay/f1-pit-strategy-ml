"""Shared helpers for the multi-track Streamlit dashboard.

All loaders degrade gracefully: if an artifact is missing they return an empty
DataFrame / ``None`` instead of raising, so the app stays usable before the
modeling steps have been run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Make the project importable from any page (so `import src...` works under
# Streamlit, whose working directory is the app folder).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "global"

YEAR = 2025
LAPS_PATH = PROCESSED_DIR / str(YEAR) / "laps_all_tracks.csv"
MODEL_READY_PATH = PROCESSED_DIR / str(YEAR) / "model_ready_all_tracks.csv"


@st.cache_data(show_spinner=False)
def safe_read_csv(path_str: str) -> pd.DataFrame:
    """Read a CSV if it exists, else return an empty DataFrame."""
    path = Path(path_str)
    if path.is_file():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_laps() -> pd.DataFrame:
    """Load the combined all-track laps dataset."""
    return safe_read_csv(str(LAPS_PATH))


@st.cache_data(show_spinner=False)
def load_model_ready() -> pd.DataFrame:
    """Load the model-ready (feature-engineered) dataset."""
    return safe_read_csv(str(MODEL_READY_PATH))


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained global model, or None if not present."""
    path = MODEL_DIR / "pit_strategy_model.pkl"
    if path.is_file():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load model metadata JSON, or empty dict if not present."""
    path = MODEL_DIR / "model_metadata.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def sidebar_filters(df: pd.DataFrame) -> dict:
    """Render sidebar filters for the given laps frame and return selections."""
    st.sidebar.header("Filters")
    filters: dict = {}
    if df.empty:
        st.sidebar.info("No data loaded yet.")
        return filters

    def _multiselect(label, col):
        if col in df.columns:
            opts = sorted(x for x in df[col].dropna().unique())
            return st.sidebar.multiselect(label, opts)
        return []

    filters["year"] = _multiselect("Year", "year")
    filters["race_name"] = _multiselect("Race", "race_name")
    filters["driver"] = _multiselect("Driver", "driver")
    filters["team"] = _multiselect("Team", "team")
    filters["compound"] = _multiselect("Tyre compound", "compound")
    filters["stint"] = _multiselect("Stint", "stint")
    filters["track_status"] = _multiselect("Track status", "track_status")
    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply non-empty multiselect filters to the frame."""
    if df.empty or not filters:
        return df
    out = df
    for col, selected in filters.items():
        if selected and col in out.columns:
            out = out[out[col].isin(selected)]
    return out


def dataset_status() -> dict:
    """Summarize which artifacts are present (for the overview page)."""
    return {
        "laps": LAPS_PATH.is_file(),
        "model_ready": MODEL_READY_PATH.is_file(),
        "model": (MODEL_DIR / "pit_strategy_model.pkl").is_file(),
        "metadata": (MODEL_DIR / "model_metadata.json").is_file(),
    }


# --------------------------------------------------------------------------- #
# Theme: dark "liquid glass" with F1-red accents
# --------------------------------------------------------------------------- #
# Palette (kept in sync with src/plotting.py).
BG = "#0E1117"
SURFACE = "#161A23"
INK = "#ECEDEE"
MUTED = "#9BA1AC"
ACCENT = "#E10600"
ACCENT_SOFT = "#FF4D5E"

_PLOTLY_READY = False


def _register_plotly_theme() -> None:
    """Register and default a dark, transparent Plotly template with red colorway."""
    global _PLOTLY_READY
    if _PLOTLY_READY:
        return
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates["f1glass"] = go.layout.Template(
        layout=dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=INK, family="Inter, system-ui, sans-serif"),
            colorway=[ACCENT, ACCENT_SOFT, "#F7C82E", "#3BB143", "#2E6FE0",
                      "#C01016", "#FF8A93", "#7A7F8A"],
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=56, r=24, b=48, l=24),
        )
    )
    pio.templates.default = "plotly_dark+f1glass"
    _PLOTLY_READY = True


# Custom fonts + glass surfaces + red accents. Selectors use stable Streamlit
# data-testids so they survive minor version changes.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0E1117;
  --ink: #ECEDEE;
  --muted: #9BA1AC;
  --accent: #E10600;
  --accent-soft: #FF4D5E;
  --glass: rgba(255,255,255,0.045);
  --glass-border: rgba(255,255,255,0.10);
  --glass-shadow: 0 8px 30px rgba(0,0,0,0.45);
}

html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }

/* Ambient background: deep base + soft red glows (liquid feel) */
.stApp {
  background:
    radial-gradient(1100px 600px at 88% -8%, rgba(225,6,0,0.16), transparent 60%),
    radial-gradient(900px 520px at -6% 110%, rgba(225,6,0,0.10), transparent 55%),
    linear-gradient(180deg, #0B0E14 0%, #0E1117 45%, #0B0E14 100%);
  background-attachment: fixed;
  color: var(--ink);
}

/* Hide default Streamlit chrome noise */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; text-wrap: balance; }
h1 {
  font-weight: 800;
  background: none;
}
h1::after {
  content: ""; display: block; width: 84px; height: 4px; margin-top: 0.5rem;
  border-radius: 99px;
  background: linear-gradient(90deg, var(--accent), var(--accent-soft));
  box-shadow: 0 0 18px rgba(225,6,0,0.55);
}

/* Sidebar: frosted glass panel */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(22,26,35,0.85), rgba(14,17,23,0.85));
  backdrop-filter: blur(16px) saturate(140%);
  -webkit-backdrop-filter: blur(16px) saturate(140%);
  border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebarNav"] a { border-radius: 10px; }
[data-testid="stSidebarNav"] a:hover { background: rgba(225,6,0,0.12); }

/* Metric cards as glass tiles */
[data-testid="stMetric"] {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 16px 18px;
  backdrop-filter: blur(12px) saturate(130%);
  -webkit-backdrop-filter: blur(12px) saturate(130%);
  box-shadow: var(--glass-shadow);
  transition: transform .35s cubic-bezier(0.16,1,0.3,1), border-color .35s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(225,6,0,0.45); }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 700; }
[data-testid="stMetricLabel"] p { color: var(--muted); }

/* Inputs / selects as glass */
[data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div,
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  background: var(--glass) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(8px);
}
[data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; }

/* Buttons */
.stButton > button, [data-testid="stBaseButton-secondary"] {
  background: linear-gradient(180deg, rgba(225,6,0,0.92), rgba(176,16,22,0.92));
  color: #fff; border: 1px solid rgba(255,77,94,0.5);
  border-radius: 12px; font-weight: 600;
  box-shadow: 0 6px 20px rgba(225,6,0,0.35);
  transition: transform .25s ease, box-shadow .25s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(225,6,0,0.5); }

/* Tabs, expanders, dataframes -> glass */
[data-testid="stExpander"], [data-testid="stDataFrame"], .stTabs [data-baseweb="tab-list"] {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 14px;
  backdrop-filter: blur(10px);
}
.stTabs [aria-selected="true"] { color: var(--accent-soft) !important; }

/* Plotly charts sit on glass cards */
[data-testid="stPlotlyChart"], [data-testid="stImage"] {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 10px;
  backdrop-filter: blur(10px);
  box-shadow: var(--glass-shadow);
}
[data-testid="stImage"] img { border-radius: 10px; }

/* Alerts: tone to glass instead of solid blocks */
[data-testid="stAlert"] {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  backdrop-filter: blur(8px);
}

a { color: var(--accent-soft); }

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
"""


def inject_theme() -> None:
    """Apply the dark glass theme: call once at the top of every page."""
    _register_plotly_theme()
    st.markdown(_CSS, unsafe_allow_html=True)
