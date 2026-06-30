"""F1 Pit Strategy ML — multi-track Streamlit dashboard (entry point).

Run with::

    streamlit run app/streamlit_app.py

Streamlit automatically discovers the pages in ``app/pages/``. This landing
page introduces the project and links to the analysis pages.
"""
import streamlit as st

from _shared import dataset_status, inject_theme, load_laps

st.set_page_config(page_title="F1 Pit Strategy ML", page_icon="🏎️", layout="wide")
inject_theme()

st.title("🏎️ F1 Pit Strategy ML — 2025 Season")
st.markdown(
    """
    This dashboard analyzes Formula 1 race strategy across the **2025 season** using
    [FastF1](https://docs.fastf1.dev/) data. It grew from a Monaco-only prototype into a
    reusable **multi-track** pipeline for studying pit-stop timing, tyre-stint behaviour,
    track-status impact, and race-specific strategy differences.

    **Predictions are exploratory and educational — not professional race-strategy software.**
    """
)

laps = load_laps()
status = dataset_status()

col1, col2, col3 = st.columns(3)
if not laps.empty:
    col1.metric("Races", laps["race_name"].nunique() if "race_name" in laps else 0)
    col2.metric("Lap records", f"{len(laps):,}")
    col3.metric("Drivers", laps["driver"].nunique() if "driver" in laps else 0)
else:
    st.warning(
        "Combined laps dataset not found. Run the Phase 4 builder:\n\n"
        "`python src/data/build_all_tracks_dataset.py --year 2025 --session R`"
    )

st.subheader("Use the pages in the sidebar")
st.markdown(
    """
    - **Project Overview** — dataset size, target, model summary, limitations
    - **Track Explorer** — per-race laps, tyre usage, stint & pit timing
    - **Strategy Comparison** — compare two races side by side
    - **Model Predictions** — predicted pit windows & race-level performance
    - **Explainability** — feature importance and SHAP
    """
)

with st.expander("Artifact status"):
    st.write({k: ("✅ found" if v else "— missing") for k, v in status.items()})
