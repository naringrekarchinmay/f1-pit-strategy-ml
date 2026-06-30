"""Page 2: Track Explorer — per-race lap, tyre, stint, and pit detail."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st

from _shared import inject_theme, load_laps

inject_theme()
st.title("Track Explorer")

laps = load_laps()
if laps.empty:
    st.info("Combined laps dataset not found. Run the Phase 4 builder first.")
    st.stop()

races = sorted(laps["race_name"].dropna().unique())
race = st.selectbox("Race", races, index=races.index("Monaco") if "Monaco" in races else 0)
race_df = laps[laps["race_name"] == race].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Laps", f"{len(race_df):,}")
c2.metric("Drivers", race_df["driver"].nunique())
c3.metric("Pit stops", int(race_df["pit_in_time"].notna().sum()) if "pit_in_time" in race_df else 0)
sc = int(race_df["safety_car_or_vsc_flag"].astype(bool).sum()) if "safety_car_or_vsc_flag" in race_df else 0
c4.metric("SC/VSC laps", sc)

st.subheader("Lap times by driver")
plot_df = race_df.dropna(subset=["lap_time_seconds"]) if "lap_time_seconds" in race_df else race_df
drivers = sorted(plot_df["driver"].dropna().unique())
sel = st.multiselect("Drivers", drivers, default=drivers[:5])
if sel:
    fig = px.line(
        plot_df[plot_df["driver"].isin(sel)],
        x="lap_number", y="lap_time_seconds", color="driver", markers=False,
        labels={"lap_time_seconds": "Lap time (s)", "lap_number": "Lap"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Tyre compound usage")
if "compound" in race_df:
    usage = race_df["compound"].value_counts().reset_index()
    usage.columns = ["compound", "laps"]
    st.plotly_chart(px.pie(usage, names="compound", values="laps"), use_container_width=True)

st.subheader("Stint behaviour (tyre life vs lap time)")
if {"tyre_life", "lap_time_seconds", "compound"} <= set(race_df.columns):
    st.plotly_chart(
        px.scatter(
            race_df.dropna(subset=["lap_time_seconds"]),
            x="tyre_life", y="lap_time_seconds", color="compound",
            hover_data=["driver", "stint", "lap_number"],
            labels={"lap_time_seconds": "Lap time (s)", "tyre_life": "Tyre life (laps)"},
        ),
        use_container_width=True,
    )

st.subheader("Pit stop timing")
if "pit_in_time" in race_df:
    pits = race_df[race_df["pit_in_time"].notna()][["driver", "lap_number", "stint", "compound"]]
    if pits.empty:
        st.write("No pit-in events detected for this race.")
    else:
        st.plotly_chart(
            px.histogram(pits, x="lap_number", nbins=20,
                         labels={"lap_number": "Pit-in lap"}),
            use_container_width=True,
        )
        st.dataframe(pits.sort_values("lap_number"), use_container_width=True)

st.subheader("Track status events")
if "track_status" in race_df:
    ts = race_df["track_status"].astype(str).value_counts().reset_index()
    ts.columns = ["track_status", "laps"]
    st.dataframe(ts, use_container_width=True)
    st.caption("FastF1 codes: 1=clear, 2=yellow, 4=safety car, 5=red, 6/7=VSC (may be concatenated).")
