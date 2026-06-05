# F1 Pit Strategy ML

A Python machine learning and race analytics project focused on Formula 1 race pace, tyre degradation, and lap-time prediction.

## Project Overview

This project uses Formula 1 race data to explore how machine learning can be used to understand race pace, tyre degradation, and lap-time prediction.

The current version focuses on the **2025 Monaco Grand Prix** using FastF1 data.

Monaco was selected as the starting race because it is one of the most strategy-sensitive circuits on the Formula 1 calendar. Since overtaking is difficult, lap time, tyre life, track position, pit timing, traffic, and race context can strongly influence race outcomes.

## Current Scope

The first version of this project focuses on:

* Collecting Monaco 2025 race data using FastF1
* Cleaning lap-level race data
* Exploring race pace by driver and compound
* Creating a stricter tyre analysis dataset
* Estimating tyre degradation by compound, driver, and stint
* Engineering machine-learning features
* Training lap-time prediction models
* Building an interactive Streamlit dashboard

## Main ML Question

Can machine learning predict Monaco 2025 race lap times using driver, team, tyre compound, tyre age, stint, lap number, race progress, track status, pit-lap status, and recent pace features?

## Project Phases

### Phase 1: Data Collection

Loaded the 2025 Monaco Grand Prix race session using FastF1 and saved raw lap, weather, and result data.

### Phase 2: Data Cleaning and EDA

Converted lap times into seconds, identified pit laps, cleaned lap-level data, removed abnormal laps for exploratory analysis, and created charts for race pace, tyre compound usage, tyre life, and driver consistency.

### Phase 3: Tyre Degradation Analysis

Estimated tyre degradation using the relationship between tyre life and lap time.

The analysis focused mainly on green-flag laps to reduce distortion from abnormal race conditions such as yellow flags, VSC periods, pit laps, and mixed track-status laps.

### Phase 4: Feature Engineering for Machine Learning

Created a machine-learning-ready dataset for Monaco 2025 lap-time prediction.

Engineered features included:

* Race progress
* Tyre life
* Tyre life squared
* Stint progress
* Driver median pace
* Team median pace
* Previous lap time
* Rolling 3-lap average
* Rolling 5-lap average
* Pit-lap indicator
* Green-flag indicator
* Track status

The final ML dataset was saved as:

`data/processed/2025_monaco_ml_dataset.csv`

### Phase 5: Model Training and Comparison

Trained and compared multiple regression models to predict `LapTimeSeconds`.

Models tested:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

The best-performing model was the **Random Forest Regressor**.

| Model             |   MAE |  RMSE |    R² |  MAPE |
| ----------------- | ----: | ----: | ----: | ----: |
| Random Forest     | 0.624 | 0.953 | 0.974 | 0.786 |
| Gradient Boosting | 0.674 | 1.034 | 0.969 | 0.844 |
| Linear Regression | 2.260 | 3.815 | 0.575 | 2.724 |

The Random Forest model predicted Monaco 2025 lap times within approximately **0.62 seconds on average** on the test set.

Important note: this first model uses a random train/test split within one race, so the results are useful for a first learning version but may be optimistic. A stronger future validation approach will train on multiple races and test on an unseen race.

### Phase 6: Streamlit Dashboard

Built an interactive Streamlit dashboard with pages for:

* Project overview
* Race overview
* Driver pace comparison
* Tyre degradation analysis
* Model performance
* Lap-time prediction

The dashboard uses the trained Random Forest model to estimate Monaco 2025 lap times based on driver, team, tyre compound, tyre age, lap number, stint, position, track status, pit-lap status, and recent pace features.

## Early Findings

* Monaco 2025 showed mild tyre degradation overall.
* Hard tyres showed the clearest positive degradation trend.
* Medium tyres were almost flat in degradation trend.
* Soft tyre results should be interpreted carefully because there were fewer soft-tyre laps.
* Driver and stint-level degradation should be interpreted carefully because Monaco lap times are strongly affected by traffic, track position, fuel burn, and race pace management.
* Random Forest performed better than Linear Regression, suggesting that lap-time prediction has non-linear patterns.
* Race progress, lap number, previous lap time, pit-lap status, stint progress, and track status were among the most important model features.

## Dashboard Preview

The Streamlit app includes:

1. **Project Overview**
   Summary of the project, model comparison, and current project status.

2. **Race Overview**
   High-level Monaco 2025 race dataset summary, average pace by driver, and compound usage.

3. **Driver Pace**
   Interactive comparison of selected drivers across the race.

4. **Tyre Degradation**
   Tyre life vs lap time analysis, compound degradation, and driver-level degradation trends.

5. **Model Performance**
   Model comparison, actual vs predicted lap times, and prediction error distribution.

6. **Lap Time Predictor**
   Manual input interface that uses the trained Random Forest model to predict expected lap time.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/naringrekarchinmay/f1-pit-strategy-ml.git
cd f1-pit-strategy-ml
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```text
f1-pit-strategy-ml/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── monaco_2025_lap_time_model.pkl
│   └── monaco_2025_model_pipeline.pkl
│
├── notebooks/
│   ├── 01_data_collection_fastf1.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_tyre_degradation_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_training.ipynb
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
├── src/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Tools Used

* Python
* FastF1
* pandas
* NumPy
* Matplotlib
* Plotly
* scikit-learn
* Streamlit
* joblib

## Current Limitations

This is a first working version of the project, so there are some important limitations:

* The current model is trained and tested only on Monaco 2025.
* The model uses a random train/test split within the same race.
* Weather features were not merged into the first ML dataset.
* The lap-time predictor is a live-race style estimator because it uses previous lap and rolling pace features.
* The model should not be treated as an official F1 strategy tool.
* Driver-level tyre degradation should be interpreted carefully because lap times are affected by traffic, fuel burn, race management, and track position.

## Next Phases

Planned future improvements:

* Add more 2025 races
* Create a multi-race ML dataset
* Compare tyre degradation across circuits
* Train on several races and test on an unseen race
* Add weather features
* Improve the Streamlit predictor with driver-team auto-matching
* Add clearer track-status labels
* Build a pit strategy recommendation model
* Explore race simulation and strategy comparison

## Future Research Direction

The long-term goal is to move from a single-race analysis project into a broader F1 strategy analytics system.

Future versions may answer questions such as:

* Which tracks show the highest tyre degradation?
* Can a model trained on previous races predict lap times at a new race?
* How does tyre degradation differ by compound and circuit?
* When does pitting become strategically beneficial?
* Can machine learning support pit-window and strategy decisions?
