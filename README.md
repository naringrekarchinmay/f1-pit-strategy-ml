# F1 Pit Strategy ML — 2025 Season

This project analyzes Formula 1 race strategy across the **2025 season** using
[FastF1](https://docs.fastf1.dev/) data. It expands from a Monaco-focused
prototype into a reusable **multi-track machine learning pipeline** for studying
pit-stop timing, tyre-stint behaviour, track-status impact, and race-specific
strategy differences.

> ⚠️ **Honest scope:** the predictions here are **exploratory and educational**.
> This is not professional race-strategy software, and the model is a modest
> baseline (see [Limitations](docs/limitations.md)).

## Problem statement & motivation

Pit strategy is one of the biggest levers a team controls during a race. *When*
a driver pits — driven by tyre degradation, stint length, track position, and
safety-car/VSC windows — can decide the result. This project asks a focused,
learnable version of that question:

> **Can we predict, from information available *up to the current lap*, whether a
> driver will pit within the next 3 laps?** (target: `will_pit_next_3_laps`)

## Project evolution: Monaco → all tracks

- **Original case study (preserved):** a Monaco 2025 lap-time analysis with
  notebooks, tyre-degradation study, and a Monaco lap-time model. These files
  remain in `notebooks/`, `data/raw/2025_monaco_*`, and `models/monaco_2025_*`.
- **Now:** a reusable pipeline that loads *any* 2025 race into a standardized
  schema, builds a combined dataset, engineers leakage-safe features, trains a
  **race-aware** pit-prediction model, and surfaces everything in a multi-page
  dashboard.

## Data source

[FastF1](https://docs.fastf1.dev/) (lap timing, tyre/stint, pit, track-status,
and weather data). Races are resolved to an **exact calendar round** to avoid
ambiguous name matching (an early bug had mapped "Great Britain" to the Austrian
GP — see [modeling notes](docs/modeling_notes.md)).

## Project workflow

See [docs/project_workflow.md](docs/project_workflow.md) for the full diagram.

```text
FastF1 → load_fastf1_data → build_all_tracks_dataset → build_features
       → train_global_model → {track_comparison, explain_model} → Streamlit
```

## Repository structure

```text
f1-pit-strategy-ml/
├── app/
│   ├── streamlit_app.py            # multi-track dashboard entry
│   ├── _shared.py                  # cached loaders + sidebar filters
│   └── pages/                      # 5 dashboard pages
├── data/
│   ├── raw/                        # FastF1 raw (incl. original Monaco CSVs)
│   └── processed/2025/
│       ├── <race>/race_laps.csv    # per-race standardized laps + weather.csv
│       ├── laps_all_tracks.csv     # combined all-track laps
│       └── model_ready_all_tracks.csv
├── models/
│   ├── global/                     # pit_strategy_model.pkl + metadata
│   └── monaco_2025_*.pkl           # original Monaco case study
├── notebooks/                      # original Monaco notebooks (01–05)
├── outputs/
│   ├── figures/{track_comparison,explainability}/ + global model figures
│   └── reports/                    # metrics + markdown summaries
├── src/
│   ├── data/                       # load_fastf1_data, build_all_tracks_dataset
│   ├── features/                   # build_features
│   ├── models/                     # train_global_model, explain_model
│   └── analysis/                   # track_comparison
├── tests/                          # pytest suite
└── docs/                           # data dictionary, workflow, modeling, limitations
```

## Dataset structure

The standardized lap schema (29 columns) is documented in
[docs/data_dictionary.md](docs/data_dictionary.md). The combined 2025 dataset
(`data/processed/2025/laps_all_tracks.csv`) covers **24 races, ~26,700 laps, 21
drivers, 10 teams**.

## Feature engineering

`src/features/build_features.py` turns the combined laps into a model-ready
dataset with **leakage-safe** features only (race context, position, tyre/stint,
past-only pit history, track-status flags). See
[docs/modeling_notes.md](docs/modeling_notes.md) for the full feature list and
the columns explicitly excluded to prevent leakage.

## Modeling approach & race-aware validation

Baselines: **Logistic Regression**, **Random Forest**, **Gradient Boosting**.

**Validation is race-aware:** we use `GroupShuffleSplit` on `race_name`, holding
*entire races* out of training. We deliberately avoid plain random row splitting,
which leaks within-race correlations and inflates scores. Headline numbers are
reported on held-out races only.

### Results (held-out races)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.72 | 0.16 | 0.50 | **0.24** | 0.70 |
| Random Forest | 0.90 | 0.25 | 0.07 | 0.11 | 0.69 |
| Gradient Boosting | 0.89 | 0.22 | 0.07 | 0.11 | 0.72 |

The target is imbalanced (~8.5% positive), so high tree-model accuracy mostly
reflects the majority class. F1/recall show the real difficulty — these are
honest, modest baselines, not a solved problem.

## Key findings from track comparison

- **Most pit stops per driver:** Australia (~4.8 — a wet, chaotic race), Canada
  (~4.2), Spain (~2.9).
- **Longest average stints:** Singapore (~28.6 laps), Hungary (~27.9), Mexico
  City (~26.9).
- Compound usage and pit-timing distributions vary strongly by circuit; see
  [outputs/reports/track_comparison_summary.md](outputs/reports/track_comparison_summary.md)
  and figures in `outputs/figures/track_comparison/`.

## Explainability summary

Permutation importance ranks **race-progress** features highest — `lap_percentage`,
`lap_number`, `stint`, `pit_stop_count_so_far`, `total_laps`. Safety-car/VSC
flags contribute weakly for this target. Full write-up:
[outputs/reports/model_explainability_summary.md](outputs/reports/model_explainability_summary.md).

## Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Pages: **Project Overview**, **Track Explorer**, **Strategy Comparison**,
**Model Predictions**, **Explainability**. The app degrades gracefully — if an
artifact (model, metrics, figures) is missing it shows guidance instead of
crashing.

## How to run the pipeline

```bash
pip install -r requirements.txt

# Build per-race + combined laps
python src/data/load_fastf1_data.py --year 2025 --race Monaco --session R
python src/data/build_all_tracks_dataset.py --year 2025 --session R

# Feature engineering + training
python src/features/build_features.py --year 2025 --target-window 3
python src/models/train_global_model.py --year 2025 --target will_pit_next_3_laps

# Analysis + explainability
python src/analysis/track_comparison.py --year 2025
python src/models/explain_model.py --year 2025

# Dashboard
streamlit run app/streamlit_app.py
```

## How to run tests

```bash
pytest
```

The suite covers race-name normalization, the standardized schema, feature
engineering (target logic, track-status flags, pit history, leakage guards),
race-aware splitting, model fitting on a small sample, and data-quality checks on
the generated datasets.

## Limitations

Summarized in [docs/limitations.md](docs/limitations.md). Highlights: modest
predictive performance, an imbalanced target, weather not yet used as a feature,
pit detection dependent on FastF1 timing, and a single season with no
cross-season validation. **Not** strategy software.

## Future improvements

- Join weather into the model-ready features.
- Cross-season training/validation.
- Calibrated probabilities and threshold tuning for the imbalanced target.
- Per-track or track-group models, and a pit-window recommendation layer.
- Richer dashboard prediction views.

## Tools

Python · FastF1 · pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib ·
Plotly · Streamlit · joblib · pytest
