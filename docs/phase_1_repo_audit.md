# Phase 1: Repository Audit

**Date:** 2026-06-29
**Branch:** `feature/multi-track-data-pipeline`
**Purpose:** Understand the current Monaco-only prototype before refactoring it into a reusable multi-track (all 2025 races) FastF1 data pipeline. This phase is documentation only — no functional code changes were made.

---

## 1. Current Repo Structure Summary

```text
f1-pit-strategy-ml/
├── app/
│   └── streamlit_app.py                # Monaco dashboard
├── data/
│   ├── cache/                          # FastF1 cache (~103 MB, gitignored)
│   ├── raw/                            # 3 Monaco raw CSVs
│   └── processed/                      # 11 Monaco processed CSVs
├── models/
│   ├── monaco_2025_lap_time_model.pkl  # gitignored (*.pkl)
│   └── monaco_2025_model_pipeline.pkl  # gitignored (*.pkl)
├── notebooks/
│   ├── 01_data_collection_fastf1.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_tyre_degradation_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_training.ipynb
├── outputs/
│   ├── figures/                        # 14 Monaco PNGs
│   └── metrics/                        # 2 Monaco metric CSVs
├── src/                                # empty (no package yet)
├── requirements.txt
├── README.md
└── .gitignore
```

The workflow is notebook-driven: collection → cleaning/EDA → degradation analysis → feature engineering → model training, with the Streamlit app consuming the processed CSVs and the saved model. There is no reusable Python package under `src/` yet.

---

## 2. Existing Monaco-Specific Files

**Notebooks (all Monaco-hard-coded):**
- `notebooks/01_data_collection_fastf1.ipynb` — `YEAR=2025`, `RACE_NAME="Monaco"`, saves `2025_monaco_*` raw CSVs.
- `notebooks/02_data_cleaning_eda.ipynb`, `03_tyre_degradation_analysis.ipynb`, `04_feature_engineering.ipynb`, `05_model_training.ipynb`.

**Raw data (`data/raw/`):**
- `2025_monaco_race_laps.csv`, `2025_monaco_weather.csv`, `2025_monaco_results.csv`.

**Processed data (`data/processed/`):**
- `2025_monaco_clean_laps.csv`, `2025_monaco_racing_laps_filtered.csv`, `2025_monaco_tyre_analysis_laps.csv`, `2025_monaco_ml_dataset.csv`, `2025_monaco_prediction_results.csv`, `2025_monaco_stint_summary.csv`, and the degradation summaries (`compound`, `driver`, `tyre`, plus `highest`/`lowest` degradation stints).

**Models (`models/`):**
- `monaco_2025_lap_time_model.pkl`, `monaco_2025_model_pipeline.pkl`.

**Outputs:**
- `outputs/metrics/monaco_2025_model_comparison.csv`, `outputs/metrics/monaco_2025_feature_importance.csv`.
- `outputs/figures/monaco_2025_*.png` (14 figures).

**Dashboard:**
- `app/streamlit_app.py` — hard-codes Monaco file paths (`2025_monaco_clean_laps.csv`, etc.) and the Monaco model.

---

## 3. Hard-Coded Values That Need Refactoring

| Location | Hard-coded value | Concern |
|---|---|---|
| `notebooks/01_data_collection_fastf1.ipynb` | `YEAR=2025`, `RACE_NAME="Monaco"`, `SESSION_TYPE="R"` | Should be parameters in a reusable loader. |
| Notebooks (all) | `Path("../data/raw")` etc. (relative to notebook cwd) | Breaks when run from a different directory; module code should resolve paths from project root. |
| Raw/processed filenames | `2025_monaco_*` flat-prefix naming | Does not scale; needs `data/<stage>/<year>/<race>/` layout. |
| `app/streamlit_app.py` | `CLEAN_LAPS_PATH = .../2025_monaco_clean_laps.csv`, `MODEL_PATH = .../monaco_2025_lap_time_model.pkl` | Tightly bound to Monaco (dashboard redesign is out of scope for Phase 1–4, noted only). |
| Column names | FastF1 PascalCase (`LapTime`, `TyreLife`, `TrackStatus`) | Target schema is snake_case and standardized across races. |

`app/streamlit_app.py` correctly uses `Path(__file__).resolve().parents[1]` for its base dir — that pattern should be reused in the new modules.

---

## 4. Current Data Files

- **Raw (`data/raw/`, ~484 KB):** Monaco laps / weather / results (FastF1-native columns).
  - Laps columns: `Time, Driver, DriverNumber, LapTime, LapNumber, Stint, PitOutTime, PitInTime, Sector1Time..3Time, ... Compound, TyreLife, FreshTyre, Team, LapStartTime, LapStartDate, TrackStatus, Position, Deleted, ... IsAccurate`.
  - Weather columns: `Time, AirTemp, Humidity, Pressure, Rainfall, TrackTemp, WindDirection, WindSpeed`.
- **Processed (`data/processed/`, ~1.6 MB):** 11 cleaned/engineered Monaco CSVs. The ML dataset (`2025_monaco_ml_dataset.csv`) columns: `Driver, Team, Compound, LapNumber, TyreLife, TyreLifeSquared, Stint, Position, TrackStatus, IsPitLap, IsGreenFlag, RaceProgress, StintLength, StintProgress, DriverMedianPace, TeamMedianPace, PreviousLapTime, Rolling3LapAvg, Rolling5LapAvg, LapTimeSeconds`.
- **Cache (`data/cache/`, ~103 MB):** FastF1 pickle cache for the Monaco race (gitignored).

---

## 5. Current Model / Output Files

- **Models:** `monaco_2025_lap_time_model.pkl`, `monaco_2025_model_pipeline.pkl` (gitignored via `*.pkl`).
- **Metrics:** `monaco_2025_model_comparison.csv`, `monaco_2025_feature_importance.csv`.
- **Figures:** 14 Monaco PNGs (pace, degradation, prediction error, feature importance, etc.).

These are downstream of the data pipeline and are **out of scope** for Phase 1–4 (no model training / explainability / dashboard redesign yet). They must be preserved.

---

## 6. Current Dashboard / App Files

- `app/streamlit_app.py` — Streamlit dashboard reading the Monaco processed CSVs, metrics, and saved model. Uses a robust `BASE_DIR` path pattern but every data path is Monaco-specific. **No changes in Phase 1–4** (dashboard redesign explicitly deferred).

---

## 7. Recommended Changes for Phase 2–4

**Phase 2 (architecture):**
- Add year/race-organized folders: `data/raw/2025/`, `data/processed/2025/`, `outputs/{figures,reports,predictions}/`, `src/{data,features,models}/`, `tests/`.
- Add a `docs/data_dictionary.md` describing the standardized snake_case schema.
- Preserve all existing Monaco files in place; new layout lives alongside them.

**Phase 3 (reusable loader — `src/data/load_fastf1_data.py`):**
- Parameterize `(year, race_name, session_type)`; resolve paths from project root.
- Standardize to a snake_case schema, derive `lap_time_seconds`, `is_pit_lap`, `pit_stop_count`, `safety_car_or_vsc_flag`, and attach metadata (`round_number`, `track_name`, `source`, `source_file`, `created_at`).
- Fill missing FastF1 fields with NA rather than crashing; log clearly.
- Write `data/processed/<year>/<race>/race_laps.csv` and `weather.csv`.

**Phase 4 (all-races builder — `src/data/build_all_tracks_dataset.py`):**
- Drive the loader across the 24-race 2025 calendar; tolerate per-race failures.
- Combine successes into `data/processed/2025/laps_all_tracks.csv` and write `outputs/reports/all_tracks_build_summary.csv`.

**Testing:** `tests/test_data_pipeline.py` covering name normalization, required columns, missing-field safety, failure tolerance, and duplicate checks.

---

## 8. Risks Before Expanding to All Tracks

1. **Event-name mismatches** — FastF1 event names differ from the spec's display names (e.g. "Great Britain" → "British Grand Prix", "São Paulo" → "São Paulo Grand Prix", "Emilia Romagna", "Mexico City"). `fastf1.get_session` is fairly forgiving with partial names, but the loader must handle/normalize names and not assume exact matches.
2. **Long load times & rate limiting** — Loading 24 races pulls a large amount of telemetry; first run is slow and may hit Ergast/F1 API throttling. Caching is essential; the build must be resumable and failure-tolerant.
3. **Missing / partial fields per race** — Not every session has complete weather, sector times, or track-status data. Extraction must guard each optional column.
4. **Schema drift** — FastF1 column availability can vary by season/session; standardize to a fixed column list and fill gaps with NA.
5. **Cache & disk size** — Monaco alone is ~103 MB of cache; 24 races will be substantially larger. Cache stays gitignored.
6. **Future-dated / unavailable races** — Some 2025 rounds may not be available depending on data source state; these should be logged as failures and skipped, not abort the build.
7. **Time zone / timedelta handling** — Lap and sector times are pandas timedeltas; converting to seconds consistently avoids downstream type errors.
