import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="F1 Pit Strategy ML",
    page_icon="🏎️",
    layout="wide"
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
MODELS_DIR = BASE_DIR / "models"

CLEAN_LAPS_PATH = DATA_DIR / "2025_monaco_clean_laps.csv"
TYRE_ANALYSIS_PATH = DATA_DIR / "2025_monaco_tyre_analysis_laps.csv"
COMPOUND_DEG_PATH = DATA_DIR / "2025_monaco_compound_degradation_summary.csv"
DRIVER_DEG_PATH = DATA_DIR / "2025_monaco_driver_degradation_summary.csv"
STINT_DEG_PATH = DATA_DIR / "2025_monaco_tyre_degradation_summary.csv"
ML_DATASET_PATH = DATA_DIR / "2025_monaco_ml_dataset.csv"
PREDICTION_RESULTS_PATH = DATA_DIR / "2025_monaco_prediction_results.csv"
MODEL_COMPARISON_PATH = METRICS_DIR / "monaco_2025_model_comparison.csv"
MODEL_PATH = MODELS_DIR / "monaco_2025_lap_time_model.pkl"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_model(path):
    if path.exists():
        return joblib.load(path)
    return None


clean_laps = load_csv(CLEAN_LAPS_PATH)
tyre_laps = load_csv(TYRE_ANALYSIS_PATH)
compound_deg = load_csv(COMPOUND_DEG_PATH)
driver_deg = load_csv(DRIVER_DEG_PATH)
stint_deg = load_csv(STINT_DEG_PATH)
ml_dataset = load_csv(ML_DATASET_PATH)
prediction_results = load_csv(PREDICTION_RESULTS_PATH)
model_comparison = load_csv(MODEL_COMPARISON_PATH)
model = load_model(MODEL_PATH)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def seconds_to_lap_time(seconds):
    """Convert seconds into M:SS.mmm format."""
    if pd.isna(seconds):
        return "N/A"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:06.3f}"


def safe_mean(series):
    if series.empty:
        return np.nan
    return series.mean()


def page_header(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.markdown(subtitle)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("F1 Pit Strategy ML")
st.sidebar.markdown("### Monaco 2025 Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Race Overview",
        "Driver Pace",
        "Tyre Degradation",
        "Model Performance",
        "Lap Time Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Current scope: **2025 Monaco GP**")
st.sidebar.markdown("Future scope: multi-race F1 strategy ML")


# ---------------------------------------------------------
# Page 1: Project Overview
# ---------------------------------------------------------

if page == "Project Overview":
    page_header(
        "F1 Pit Strategy ML",
        "A Python machine learning project analyzing Formula 1 race pace, tyre degradation, and lap-time prediction."
    )

    st.markdown(
        """
        This dashboard currently focuses on the **2025 Monaco Grand Prix**.

        Monaco is a strategy-heavy street circuit where overtaking is difficult. Because of that, lap time, tyre life,
        track position, pit timing, and race context can strongly influence race outcomes.

        This project has completed:

        - **Phase 1:** FastF1 data collection
        - **Phase 2:** Data cleaning and exploratory data analysis
        - **Phase 3:** Tyre degradation analysis
        - **Phase 4:** Feature engineering for machine learning
        - **Phase 5:** Lap-time prediction model training
        - **Phase 6:** Streamlit dashboard
        """
    )

    st.subheader("Main ML Question")
    st.info(
        "Can machine learning predict Monaco 2025 race lap times using driver, team, tyre, stint, race progress, "
        "track status, pit-lap status, and recent pace features?"
    )

    if not model_comparison.empty:
        st.subheader("Model Comparison")
        st.dataframe(model_comparison, use_container_width=True)

        best_model = model_comparison.sort_values("MAE").iloc[0]
        st.success(
            f"Best model: {best_model['Model']} with MAE of {best_model['MAE']:.3f} seconds."
        )


# ---------------------------------------------------------
# Page 2: Race Overview
# ---------------------------------------------------------

elif page == "Race Overview":
    page_header("Race Overview", "High-level summary of the Monaco 2025 race dataset.")

    if clean_laps.empty:
        st.error("Clean laps data not found.")
    else:
        total_laps = len(clean_laps)
        drivers = clean_laps["Driver"].nunique()
        teams = clean_laps["Team"].nunique()
        compounds = clean_laps["Compound"].nunique()
        avg_lap = clean_laps["LapTimeSeconds"].mean()
        fastest_lap = clean_laps["LapTimeSeconds"].min()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Lap Records", f"{total_laps:,}")
        col2.metric("Drivers", drivers)
        col3.metric("Teams", teams)
        col4.metric("Compounds", compounds)
        col5.metric("Fastest Lap", seconds_to_lap_time(fastest_lap))

        st.subheader("Average Clean Race Pace by Driver")

        pace_df = (
            clean_laps[
                (clean_laps["IsPitLap"] == False) &
                (clean_laps["LapTimeSeconds"] >= 73) &
                (clean_laps["LapTimeSeconds"] <= 83)
            ]
            .groupby("Driver")["LapTimeSeconds"]
            .mean()
            .reset_index()
            .sort_values("LapTimeSeconds")
        )

        fig = px.bar(
            pace_df,
            x="Driver",
            y="LapTimeSeconds",
            title="Average Clean Race Pace by Driver",
            labels={"LapTimeSeconds": "Average Lap Time Seconds"}
        )
        fig.update_layout(yaxis_range=[73, 82])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Compound Usage")
        compound_counts = clean_laps["Compound"].value_counts().reset_index()
        compound_counts.columns = ["Compound", "LapCount"]

        fig2 = px.pie(
            compound_counts,
            names="Compound",
            values="LapCount",
            title="Lap Count by Tyre Compound"
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------
# Page 3: Driver Pace
# ---------------------------------------------------------

elif page == "Driver Pace":
    page_header("Driver Pace Analysis", "Compare lap-time trends by driver.")

    if clean_laps.empty:
        st.error("Clean laps data not found.")
    else:
        drivers = sorted(clean_laps["Driver"].dropna().unique())

        selected_drivers = st.multiselect(
            "Select drivers",
            drivers,
            default=[d for d in ["NOR", "LEC", "PIA", "VER"] if d in drivers]
        )

        lap_filter = st.slider(
            "Lap-time range for chart",
            min_value=float(clean_laps["LapTimeSeconds"].min()),
            max_value=float(clean_laps["LapTimeSeconds"].max()),
            value=(73.0, 83.0)
        )

        pace_data = clean_laps[
            (clean_laps["Driver"].isin(selected_drivers)) &
            (clean_laps["LapTimeSeconds"] >= lap_filter[0]) &
            (clean_laps["LapTimeSeconds"] <= lap_filter[1])
        ].copy()

        if pace_data.empty:
            st.warning("No data available for selected filters.")
        else:
            fig = px.line(
                pace_data,
                x="LapNumber",
                y="LapTimeSeconds",
                color="Driver",
                markers=True,
                title="Lap Time by Driver",
                labels={"LapTimeSeconds": "Lap Time Seconds"}
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Driver Pace Summary")

            summary = (
                pace_data
                .groupby("Driver")["LapTimeSeconds"]
                .agg(["count", "mean", "median", "std", "min"])
                .reset_index()
                .sort_values("mean")
            )

            st.dataframe(summary, use_container_width=True)


# ---------------------------------------------------------
# Page 4: Tyre Degradation
# ---------------------------------------------------------

elif page == "Tyre Degradation":
    page_header("Tyre Degradation Analysis", "Explore tyre life, compounds, and degradation trends.")

    if tyre_laps.empty:
        st.error("Tyre analysis data not found.")
    else:
        st.subheader("Tyre Life vs Lap Time")

        selected_compounds = st.multiselect(
            "Select compounds",
            sorted(tyre_laps["Compound"].dropna().unique()),
            default=sorted(tyre_laps["Compound"].dropna().unique())
        )

        filtered_tyre = tyre_laps[tyre_laps["Compound"].isin(selected_compounds)].copy()

        fig = px.scatter(
            filtered_tyre,
            x="TyreLife",
            y="LapTimeSeconds",
            color="Compound",
            hover_data=["Driver", "LapNumber", "Stint"],
            title="Tyre Life vs Lap Time",
            labels={"LapTimeSeconds": "Lap Time Seconds"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Compound Degradation Summary")

        if not compound_deg.empty:
            st.dataframe(compound_deg, use_container_width=True)

            fig2 = px.bar(
                compound_deg.sort_values("EstimatedDegPerLapSeconds", ascending=False),
                x="Compound",
                y="EstimatedDegPerLapSeconds",
                title="Estimated Degradation by Compound",
                labels={
                    "EstimatedDegPerLapSeconds": "Estimated Seconds Lost per Tyre Lap"
                }
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Driver Degradation Summary")

        if not driver_deg.empty:
            fig3 = px.bar(
                driver_deg.sort_values("EstimatedDegPerLapSeconds"),
                x="Driver",
                y="EstimatedDegPerLapSeconds",
                title="Estimated Degradation by Driver",
                labels={
                    "EstimatedDegPerLapSeconds": "Estimated Lap-Time Change per Tyre Lap"
                }
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.caption(
                "Driver-level degradation should be interpreted carefully because Monaco lap times are affected by traffic, "
                "track position, fuel burn, and race context."
            )


# ---------------------------------------------------------
# Page 5: Model Performance
# ---------------------------------------------------------

elif page == "Model Performance":
    page_header("Model Performance", "Evaluate Monaco 2025 lap-time prediction models.")

    if model_comparison.empty:
        st.error("Model comparison file not found.")
    else:
        st.subheader("Model Comparison")
        st.dataframe(model_comparison, use_container_width=True)

        best_model = model_comparison.sort_values("MAE").iloc[0]
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Best Model", best_model["Model"])
        col2.metric("MAE", f"{best_model['MAE']:.3f} sec")
        col3.metric("RMSE", f"{best_model['RMSE']:.3f} sec")
        col4.metric("R²", f"{best_model['R2']:.3f}")

    if not prediction_results.empty:
        st.subheader("Actual vs Predicted Lap Times")

        fig = px.scatter(
            prediction_results,
            x="ActualLapTimeSeconds",
            y="PredictedLapTimeSeconds",
            color="Driver",
            hover_data=["Driver", "Team", "Compound", "LapNumber", "IsPitLap", "TrackStatus"],
            title="Actual vs Predicted Lap Times"
        )

        min_val = min(
            prediction_results["ActualLapTimeSeconds"].min(),
            prediction_results["PredictedLapTimeSeconds"].min()
        )
        max_val = max(
            prediction_results["ActualLapTimeSeconds"].max(),
            prediction_results["PredictedLapTimeSeconds"].max()
        )

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(dash="dash")
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction Error Distribution")

        fig2 = px.histogram(
            prediction_results,
            x="PredictionErrorSeconds",
            nbins=30,
            title="Prediction Error Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            """
            **Interpretation:**  
            Positive error means the actual lap was slower than predicted.  
            Negative error means the actual lap was faster than predicted.
            """
        )


# ---------------------------------------------------------
# Page 6: Lap Time Predictor
# ---------------------------------------------------------

elif page == "Lap Time Predictor":
    page_header("Lap Time Predictor", "Use the trained Random Forest model to predict Monaco 2025 lap time.")

    if model is None:
        st.error("Model file not found. Please run Phase 5 first.")
    elif ml_dataset.empty:
        st.error("ML dataset not found.")
    else:
        st.markdown(
            """
            This is a **live-race style predictor** because it uses previous lap and rolling pace features.
            It is not yet a pre-race strategy simulator.
            """
        )

        drivers = sorted(ml_dataset["Driver"].dropna().unique())
        teams = sorted(ml_dataset["Team"].dropna().unique())
        compounds = sorted(ml_dataset["Compound"].dropna().unique())

        col1, col2, col3 = st.columns(3)

        with col1:
            driver = st.selectbox("Driver", drivers, index=drivers.index("NOR") if "NOR" in drivers else 0)
            team = st.selectbox("Team", teams)
            compound = st.selectbox("Compound", compounds)

        with col2:
            lap_number = st.slider("Lap Number", 1, 78, 30)
            tyre_life = st.slider("Tyre Life", 1, 75, 15)
            stint = st.slider("Stint", 1, 4, 2)

        with col3:
            position = st.slider("Position", 1, 20, 5)
            track_status = st.selectbox("Track Status", sorted(ml_dataset["TrackStatus"].unique()))
            is_pit_lap = st.selectbox("Is Pit Lap?", [0, 1])

        # Derived inputs
        max_lap = ml_dataset["LapNumber"].max()
        race_progress = lap_number / max_lap
        tyre_life_squared = tyre_life ** 2
        is_green_flag = 1 if str(track_status) == "1" else 0

        # Use historical medians/defaults from ML dataset
        driver_rows = ml_dataset[ml_dataset["Driver"] == driver]
        team_rows = ml_dataset[ml_dataset["Team"] == team]

        driver_median_pace = driver_rows["DriverMedianPace"].median()
        team_median_pace = team_rows["TeamMedianPace"].median()

        previous_lap_time = driver_rows["PreviousLapTime"].median()
        rolling3 = driver_rows["Rolling3LapAvg"].median()
        rolling5 = driver_rows["Rolling5LapAvg"].median()

        stint_length = int(ml_dataset["StintLength"].median())
        stint_progress = min(1.0, tyre_life / max(stint_length, 1))

        input_data = pd.DataFrame([{
            "Driver": driver,
            "Team": team,
            "Compound": compound,
            "LapNumber": lap_number,
            "TyreLife": tyre_life,
            "TyreLifeSquared": tyre_life_squared,
            "Stint": stint,
            "Position": position,
            "TrackStatus": track_status,
            "IsPitLap": is_pit_lap,
            "IsGreenFlag": is_green_flag,
            "RaceProgress": race_progress,
            "StintLength": stint_length,
            "StintProgress": stint_progress,
            "DriverMedianPace": driver_median_pace,
            "TeamMedianPace": team_median_pace,
            "PreviousLapTime": previous_lap_time,
            "Rolling3LapAvg": rolling3,
            "Rolling5LapAvg": rolling5
        }])

        st.subheader("Input Features")
        st.dataframe(input_data, use_container_width=True)

        if st.button("Predict Lap Time"):
            prediction = model.predict(input_data)[0]

            st.success(
                f"Predicted lap time: {seconds_to_lap_time(prediction)} "
                f"({prediction:.3f} seconds)"
            )

            st.caption(
                "This prediction is based on the Monaco 2025 model and should be treated as an educational ML estimate, "
                "not an official F1 strategy calculation."
            )