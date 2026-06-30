"""Page 4: Model Predictions — pit-window predictions and race-level performance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st

from _shared import REPORTS_DIR, load_metadata, load_model, load_model_ready, safe_read_csv

st.title("Model Predictions")

model = load_model()
data = load_model_ready()
meta = load_metadata()

if model is None or data.empty:
    st.info(
        "Trained model or model-ready dataset not found.\n\n"
        "Run:\n"
        "`python src/features/build_features.py --year 2025 --target-window 3`\n\n"
        "`python src/models/train_global_model.py --year 2025 --target will_pit_next_3_laps`"
    )
    st.stop()

features = meta.get("features", [])
st.caption(f"Model: {meta.get('best_model', 'n/a')} · target `will_pit_next_3_laps` · "
           f"{meta.get('validation', '')}")

races = sorted(data["race_name"].dropna().unique())
race = st.selectbox("Race", races, index=races.index("Monaco") if "Monaco" in races else 0)
race_df = data[data["race_name"] == race].copy()

# Rebuild the feature matrix the same way training did.
try:
    from src.models.train_global_model import select_features

    X_all, y_all, names = select_features(data)
    mask = (data["race_name"] == race).reset_index(drop=True)
    X_race = X_all[mask.values]
    proba = model.predict_proba(X_race)[:, 1]
    pred = (proba >= 0.5).astype(int)
    race_df = race_df.reset_index(drop=True)
    race_df["pit_probability"] = proba
    race_df["predicted_pit"] = pred
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not score this race: {exc}")
    st.stop()

st.subheader("Predicted pit probability by lap")
drivers = sorted(race_df["driver"].dropna().unique())
sel = st.multiselect("Drivers", drivers, default=drivers[:4])
if sel:
    st.plotly_chart(
        px.line(
            race_df[race_df["driver"].isin(sel)],
            x="lap_number", y="pit_probability", color="driver",
            labels={"pit_probability": "P(pit ≤ 3 laps)", "lap_number": "Lap"},
        ),
        use_container_width=True,
    )

st.subheader("Actual vs predicted pit windows")
if "will_pit_next_3_laps" in race_df:
    cm = race_df.groupby(["will_pit_next_3_laps", "predicted_pit"]).size().reset_index(name="count")
    st.plotly_chart(
        px.bar(cm, x="will_pit_next_3_laps", y="count", color="predicted_pit", barmode="group",
               labels={"will_pit_next_3_laps": "Actual (pit ≤3)", "predicted_pit": "Predicted"}),
        use_container_width=True,
    )

st.subheader("Driver-level prediction table")
cols = [c for c in ["driver", "lap_number", "compound", "tyre_life", "pit_probability",
                    "predicted_pit", "will_pit_next_3_laps"] if c in race_df.columns]
st.dataframe(race_df[cols].sort_values(["driver", "lap_number"]), use_container_width=True)

st.subheader("Race-level model performance (held-out test races)")
race_metrics = safe_read_csv(str(REPORTS_DIR / "race_level_model_metrics.csv"))
if race_metrics.empty:
    st.info("No race-level metrics file found.")
else:
    st.dataframe(race_metrics, use_container_width=True)
    st.caption("Metrics are reported only for races held out of training (race-aware split).")

st.caption("Predictions are exploratory/educational, not professional strategy advice.")
