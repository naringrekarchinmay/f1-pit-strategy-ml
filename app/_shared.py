"""Shared helpers for the multi-track Streamlit dashboard.

All loaders degrade gracefully: if an artifact is missing they return an empty
DataFrame / ``None`` instead of raising, so the app stays usable before the
modeling steps have been run.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
