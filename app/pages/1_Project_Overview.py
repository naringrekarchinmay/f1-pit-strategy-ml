"""Page 1: Project Overview."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from _shared import load_laps, load_metadata, load_model_ready

st.title("Project Overview")

st.markdown(
    """
    **Objective:** Study Formula 1 pit-stop strategy across the 2025 season and build an
    *exploratory* model that flags when a driver is likely to pit soon.
    """
)

laps = load_laps()
model_ready = load_model_ready()
meta = load_metadata()

st.subheader("Dataset")
if laps.empty:
    st.info("Combined laps dataset not found. Run the Phase 4 builder first.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Races", laps["race_name"].nunique())
    c2.metric("Lap records", f"{len(laps):,}")
    c3.metric("Drivers", laps["driver"].nunique())
    c4.metric("Teams", laps["team"].nunique() if "team" in laps else 0)

st.subheader("Target variable")
st.markdown(
    "`will_pit_next_3_laps` — **1** if the driver makes a pit stop within the next 3 laps, "
    "else **0**. Built leakage-safe: only current-and-past lap information is used as features."
)
if not model_ready.empty and "will_pit_next_3_laps" in model_ready:
    bal = model_ready["will_pit_next_3_laps"].value_counts(normalize=True)
    st.write(
        f"Class balance — positives: **{bal.get(1, 0):.1%}**, negatives: **{bal.get(0, 0):.1%}** "
        f"({int(model_ready['will_pit_next_3_laps'].sum()):,} positive rows of {len(model_ready):,})."
    )
else:
    st.info("Model-ready dataset not found. Run `python src/features/build_features.py`.")

st.subheader("Model summary")
if meta:
    c1, c2, c3 = st.columns(3)
    c1.metric("Best model", meta.get("best_model", "n/a"))
    c2.metric("Train races", meta.get("n_train_races", "n/a"))
    c3.metric("Test races", meta.get("n_test_races", "n/a"))
    st.caption(f"Validation: {meta.get('validation', 'n/a')}")
else:
    st.info("No trained model metadata found. Run `python src/models/train_global_model.py`.")

st.subheader("Current limitations")
st.markdown(
    """
    - Predictions are **educational/exploratory**, not professional strategy software.
    - The pit target is imbalanced; recall/precision are modest and reported honestly.
    - Weather is collected per race but **not yet joined** into the model-ready features.
    - Pit detection relies on FastF1 timing fields, which occasionally miss laps.
    """
)
