# F1 Pit Strategy ML

A Python machine learning and race analytics project focused on Formula 1 race pace, tyre degradation, and lap-time prediction.

## Current Scope

This project currently analyzes the 2025 Monaco Grand Prix using FastF1 data.

The first version focuses on:

- Collecting race data using FastF1
- Cleaning lap-level race data
- Exploring race pace by driver and compound
- Creating a stricter tyre analysis dataset
- Estimating tyre degradation by compound, driver, and stint

## Project Phases

### Phase 1: Data Collection

Loaded the 2025 Monaco Grand Prix race session using FastF1 and saved raw lap, weather, and result data.

### Phase 2: Data Cleaning and EDA

Converted lap times into seconds, identified pit laps, removed abnormal laps for analysis, and created exploratory charts for race pace, tyre compounds, and driver consistency.

### Phase 3: Tyre Degradation Analysis

Estimated tyre degradation using the relationship between tyre life and lap time. The analysis focused mainly on green-flag laps to reduce distortion from abnormal race conditions.

## Early Findings

- Monaco 2025 shows mild tyre degradation overall.
- Hard tyres showed the clearest positive degradation trend.
- Medium tyres were almost flat in degradation trend.
- Soft tyre results should be interpreted carefully because there were fewer soft-tyre laps.
- Driver and stint-level degradation should be interpreted carefully because Monaco lap times are strongly affected by traffic, track position, fuel burn, and race pace management.

## Next Phases

- Phase 4: Feature Engineering for Machine Learning
- Phase 5: Lap-Time Prediction Model Training
- Phase 6: Streamlit Dashboard
- Phase 7: Multi-Race Expansion
- Phase 8: Pit Strategy Recommendation

## Tools Used

- Python
- pandas
- NumPy
- Matplotlib
- FastF1
- scikit-learn, planned
- Streamlit, planned