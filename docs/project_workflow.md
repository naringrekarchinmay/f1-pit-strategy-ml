# Project Workflow

End-to-end flow from raw FastF1 data to the dashboard.

```text
FastF1 API
   │  (per race, resolved to exact calendar round)
   ▼
src/data/load_fastf1_data.py ──► data/processed/<year>/<race>/race_laps.csv
   │                                                         weather.csv
   ▼
src/data/build_all_tracks_dataset.py ──► data/processed/2025/laps_all_tracks.csv
   │                                       outputs/reports/all_tracks_build_summary.csv
   ▼
src/features/build_features.py ──► data/processed/2025/model_ready_all_tracks.csv
   │                                 outputs/reports/feature_engineering_summary.csv
   ▼
src/models/train_global_model.py ──► models/global/pit_strategy_model.pkl
   │                                   models/global/model_metadata.json
   │                                   outputs/reports/global_model_metrics.csv
   │                                   outputs/reports/race_level_model_metrics.csv
   │                                   outputs/figures/{confusion_matrix_global,model_comparison}.png
   ├──► src/analysis/track_comparison.py ──► outputs/figures/track_comparison/*
   │                                          outputs/reports/track_comparison_summary.md
   └──► src/models/explain_model.py ──► outputs/figures/explainability/*
                                         outputs/reports/model_explainability_summary.md
   ▼
app/streamlit_app.py (+ app/pages/*) ──► interactive multi-track dashboard
```

## Commands

```bash
# 1. Build per-race + combined laps (Phase 3–4)
python src/data/load_fastf1_data.py --year 2025 --race Monaco --session R
python src/data/build_all_tracks_dataset.py --year 2025 --session R

# 2. Feature engineering (Phase 5)
python src/features/build_features.py --year 2025 --target-window 3

# 3. Train global model with race-aware validation (Phase 6)
python src/models/train_global_model.py --year 2025 --target will_pit_next_3_laps

# 4. Analysis & explainability (Phase 7–8)
python src/analysis/track_comparison.py --year 2025
python src/models/explain_model.py --year 2025

# 5. Dashboard (Phase 9)
streamlit run app/streamlit_app.py

# Tests
pytest
```

## Phase history

- **Phase 1–4:** reusable FastF1 pipeline, multi-track dataset (Monaco → all 2025 races).
- **Phase 5:** leakage-safe feature engineering + `will_pit_next_3_laps` target.
- **Phase 6:** race-aware baseline models (LR / RF / GB).
- **Phase 7:** track comparison analysis.
- **Phase 8:** model explainability (importance, permutation, SHAP).
- **Phase 9:** multi-page Streamlit dashboard.
- **Phase 10–12:** documentation, tests, verification.
