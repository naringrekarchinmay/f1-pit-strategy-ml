"""Page 3: Strategy Comparison — compare two races side by side."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st

from _shared import load_laps

st.title("Strategy Comparison")

laps = load_laps()
if laps.empty:
    st.info("Combined laps dataset not found. Run the Phase 4 builder first.")
    st.stop()

races = sorted(laps["race_name"].dropna().unique())
c1, c2 = st.columns(2)
race_a = c1.selectbox("Race A", races, index=0)
race_b = c2.selectbox("Race B", races, index=min(1, len(races) - 1))


def _summary(df):
    pit = df[df["pit_in_time"].notna()] if "pit_in_time" in df else df[df.get("is_pit_lap", False)]
    stops_per_driver = pit.groupby("driver").size().mean() if not pit.empty else 0.0
    stint_len = df.groupby(["driver", "stint"]).size().mean() if "stint" in df else float("nan")
    sc_share = pit["safety_car_or_vsc_flag"].astype(bool).mean() if not pit.empty and "safety_car_or_vsc_flag" in pit else 0.0
    return {
        "avg_pit_stops": round(float(stops_per_driver), 2),
        "avg_stint_length": round(float(stint_len), 2),
        "pit_under_sc_vsc_share": round(float(sc_share), 2),
    }


da = laps[laps["race_name"] == race_a]
db = laps[laps["race_name"] == race_b]
sa, sb = _summary(da), _summary(db)

st.subheader("Key strategy metrics")
m1, m2, m3 = st.columns(3)
m1.metric("Avg pit stops", sa["avg_pit_stops"], delta=round(sa["avg_pit_stops"] - sb["avg_pit_stops"], 2))
m2.metric("Avg stint length", sa["avg_stint_length"], delta=round(sa["avg_stint_length"] - sb["avg_stint_length"], 2))
m3.metric("Pit under SC/VSC", sa["pit_under_sc_vsc_share"], delta=round(sa["pit_under_sc_vsc_share"] - sb["pit_under_sc_vsc_share"], 2))
st.caption(f"Deltas are {race_a} minus {race_b}.")

st.subheader("Pit-in lap distribution")
for label, df in [(race_a, da), (race_b, db)]:
    if "pit_in_time" in df:
        pits = df[df["pit_in_time"].notna()]
        if not pits.empty:
            st.plotly_chart(
                px.histogram(pits, x="lap_number", nbins=20, title=f"{label}: pit-in laps"),
                use_container_width=True,
            )

st.subheader("Compound usage")
comp = []
for label, df in [(race_a, da), (race_b, db)]:
    if "compound" in df:
        u = df["compound"].value_counts(normalize=True).reset_index()
        u.columns = ["compound", "share"]
        u["race"] = label
        comp.append(u)
if comp:
    import pandas as pd

    comp_df = pd.concat(comp, ignore_index=True)
    st.plotly_chart(
        px.bar(comp_df, x="compound", y="share", color="race", barmode="group"),
        use_container_width=True,
    )
