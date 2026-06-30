# F1 Pit Strategy ML — Phases 5–12 (Modeling, Analysis, Dashboard, Docs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the feature-engineering, modeling, track-analysis, explainability, dashboard, documentation, and testing layers on top of the existing Phase 1–4 multi-track data pipeline.

**Architecture:** Add focused, CLI-runnable modules under `src/features`, `src/models`, `src/analysis` that consume `data/processed/2025/laps_all_tracks.csv`. Train a leakage-safe binary classifier for `will_pit_next_3_laps` using **race-aware** validation (GroupShuffleSplit on `race_name`). Surface results through a multi-page Streamlit app that degrades gracefully when artifacts are missing. Keep all Phase 1–4 files and Monaco work intact.

**Tech Stack:** Python 3.13, pandas 2.2, scikit-learn 1.6, xgboost 3.2, shap 0.51, matplotlib 3.10, plotly 5.24, streamlit 1.45, joblib, pytest 8.3.

## Global Constraints

- Complete **only Phase 5–12**. Do NOT continue past Phase 12 without approval.
- Do NOT delete Monaco work or Phase 1–4 files; build on top.
- No hard-coded Monaco logic in reusable modules (track groupings in Phase 7 are analysis labels only).
- **No target leakage:** features use only current-and-past lap info. Exclude future compound, future pit laps, and any column that reveals the target. Excluded-from-features columns: `lap_time` (string), `pit_in_time`, `pit_out_time`, `pit_in_lap`, `pit_out_lap`, `source`, `source_file`, `created_at`, and the target itself.
- **Race-aware validation only** — never random row split as the sole method. Use GroupShuffleSplit with `race_name` as the group; keep whole races out of training.
- Honest naming and honest README: predictions are exploratory/educational, not professional race-strategy software. If a dataset is not fully model-ready, do not call it model-ready.
- Weather columns are NOT present in `laps_all_tracks.csv` (they live in separate per-race `weather.csv`); document them as unavailable in the model-ready set rather than fabricating a join.
- Add clear logging; keep code modular; frequent small commits.

## File Structure

- `src/features/build_features.py` — feature engineering + target + CLI.
- `src/models/train_global_model.py` — race-aware training, metrics, artifacts + CLI.
- `src/analysis/__init__.py`, `src/analysis/track_comparison.py` — track comparison figures + report + CLI.
- `src/models/explain_model.py` — feature/permutation/SHAP importance + report + CLI.
- `app/streamlit_app.py` — rewritten entry/overview; `app/_shared.py` — cached loaders & filters; `app/pages/1_Project_Overview.py`…`5_Explainability.py`.
- `tests/test_feature_engineering.py`, `tests/test_model_training.py` — new tests.
- `docs/project_workflow.md`, `docs/modeling_notes.md`, `docs/limitations.md`; `README.md` rewrite.
- Outputs: `data/processed/2025/model_ready_all_tracks.csv`, `outputs/reports/*.csv|*.md`, `models/global/*`, `outputs/figures/{track_comparison,explainability}/*`.

---

### Task 1 (Phase 5): Multi-track feature engineering

**Files:**
- Create: `src/features/build_features.py`
- Test: `tests/test_feature_engineering.py`

**Interfaces:**
- Produces:
  - `load_laps_dataset(path) -> pd.DataFrame`
  - `standardize_feature_columns(df) -> pd.DataFrame`
  - `add_race_context_features(df) -> pd.DataFrame` — adds `total_laps`, `lap_percentage`.
  - `add_tire_stint_features(df) -> pd.DataFrame` — adds `laps_since_pit`, `current_stint_lap`.
  - `add_pit_history_features(df) -> pd.DataFrame` — adds `pit_stop_count_so_far`, `has_pitted`, `laps_since_last_pit`.
  - `add_track_status_features(df) -> pd.DataFrame` — adds `is_green_flag`, `is_safety_car`, `is_vsc`.
  - `create_pit_window_target(df, window_laps=3) -> pd.DataFrame` — adds `will_pit_next_3_laps`.
  - `add_position_features(df) -> pd.DataFrame` — adds `position_change`.
  - `build_model_ready_dataset(input_path, output_path, target_window=3) -> pd.DataFrame`
  - `FEATURE_COLUMNS: list[str]`, `LEAKAGE_COLUMNS: list[str]`, `TARGET_COLUMN = "will_pit_next_3_laps"`
  - `main()` — CLI `--year --target-window`.
- Consumes: `data/processed/2025/laps_all_tracks.csv` (29-col schema from Phase 4).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_feature_engineering.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    create_pit_window_target,
    add_track_status_features,
    add_tire_stint_features,
    add_pit_history_features,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
)


def _toy():
    # One driver, one race, 6 laps, pits (pit_in) on lap 4.
    return pd.DataFrame({
        "year": [2025] * 6,
        "race_name": ["Test"] * 6,
        "driver": ["VER"] * 6,
        "lap_number": [1, 2, 3, 4, 5, 6],
        "stint": [1, 1, 1, 1, 2, 2],
        "tyre_life": [1, 2, 3, 4, 1, 2],
        "pit_in_time": [pd.NA, pd.NA, pd.NA, "0:55:00", pd.NA, pd.NA],
        "is_pit_lap": [False, False, False, True, False, False],
        "track_status": ["1", "1", "4", "1", "67", "1"],
    })


def test_target_marks_laps_before_pit():
    out = create_pit_window_target(_toy(), window_laps=3)
    # Pit-in is lap 4 -> laps 1,2,3 are within 3 laps before -> 1.
    assert out.loc[out["lap_number"] == 1, TARGET_COLUMN].iloc[0] == 1
    assert out.loc[out["lap_number"] == 3, TARGET_COLUMN].iloc[0] == 1
    # Lap 5,6 have no upcoming pit -> 0.
    assert out.loc[out["lap_number"] == 5, TARGET_COLUMN].iloc[0] == 0
    assert set(out[TARGET_COLUMN].unique()) <= {0, 1}


def test_track_status_flags():
    out = add_track_status_features(_toy())
    assert out.loc[out["lap_number"] == 3, "is_safety_car"].iloc[0] == 1
    assert out.loc[out["lap_number"] == 5, "is_vsc"].iloc[0] == 1
    assert out.loc[out["lap_number"] == 1, "is_green_flag"].iloc[0] == 1


def test_pit_history_monotonic():
    out = add_pit_history_features(_toy())
    counts = out.sort_values("lap_number")["pit_stop_count_so_far"].tolist()
    assert counts == sorted(counts)  # non-decreasing
    assert out["has_pitted"].iloc[-1] == 1


def test_no_leakage_columns_in_target_features():
    assert "pit_in_lap" in LEAKAGE_COLUMNS
    assert "pit_out_time" in LEAKAGE_COLUMNS
    assert TARGET_COLUMN not in LEAKAGE_COLUMNS
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/test_feature_engineering.py -q` → FAIL (import error).
- [ ] **Step 3: Implement `src/features/build_features.py`.** `PROJECT_ROOT = Path(__file__).resolve().parents[2]`. Logging configured. Key logic:
  - `add_race_context_features`: `total_laps = groupby([year,race_name]).lap_number.transform('max')`; `lap_percentage = lap_number/total_laps`.
  - `add_tire_stint_features`: sort by `[year,race_name,driver,lap_number]`; `current_stint_lap = groupby([year,race_name,driver,stint]).cumcount()+1`; `laps_since_pit = current_stint_lap - 1`.
  - `add_pit_history_features`: `pit_event = pit_in_time.notna()`; `pit_stop_count_so_far = groupby([year,race_name,driver]).pit_event.cumsum() - pit_event.astype(int)` (count of pits BEFORE this lap to avoid leakage); `has_pitted = (pit_stop_count_so_far>0).astype(int)`; `laps_since_last_pit` = laps since last pit-in lap (fill large/NA with `lap_number`).
  - `add_track_status_features`: cast `track_status` to str; `is_safety_car = contains '4'`, `is_vsc = contains '6' or '7'`, `is_green_flag = (no 2/4/5/6/7 codes)` i.e. status is only '1'.
  - `add_position_features`: `position_change = groupby([...driver]).position.diff().fillna(0)`.
  - `create_pit_window_target`: for each `[year,race_name,driver]`, mark `pit_event` laps; target=1 if any pit_event in `(lap, lap+window]`. Implement by, per group, for each lap L, check pit_event laps in `L+1..L+window`. Vectorized: build set of pit laps per group, then `target = any(pl in range(L+1,L+w+1))`.
  - `FEATURE_COLUMNS`: the safe numeric/categorical list (race context, tyre/stint, pit history, track status, position) — see Task 2 consumer list. `LEAKAGE_COLUMNS = ["lap_time","pit_in_time","pit_out_time","pit_in_lap","pit_out_lap","source","source_file","created_at"]`.
  - `build_model_ready_dataset`: load → standardize → add all feature groups → target → save CSV; also write `outputs/reports/feature_engineering_summary.csv` (column, dtype, group, n_missing, role).
  - `main()`: argparse `--year` (default 2025), `--target-window` (default 3); input `data/processed/<year>/laps_all_tracks.csv`, output `data/processed/<year>/model_ready_all_tracks.csv`.
- [ ] **Step 4: Run tests** → PASS (4 tests).
- [ ] **Step 5: Run CLI** — `python src/features/build_features.py --year 2025 --target-window 3` → writes model-ready CSV + summary; log target balance.
- [ ] **Step 6: Commit** — `git add src/features/build_features.py tests/test_feature_engineering.py && git commit -m "feat: add multi-track feature engineering"` (data artifacts committed in Task 9 batch or here).

---

### Task 2 (Phase 6): Train global multi-track model

**Files:**
- Create: `src/models/train_global_model.py`
- Test: `tests/test_model_training.py`

**Interfaces:**
- Produces:
  - `load_model_ready(path) -> pd.DataFrame`
  - `select_features(df) -> tuple[pd.DataFrame, pd.Series, list[str]]` — returns `(X, y, feature_names)` using safe features; one-hot encodes `compound`; drops `LEAKAGE_COLUMNS`.
  - `race_aware_split(X, y, groups, test_size=0.25, random_state=42) -> (train_idx, test_idx)` — GroupShuffleSplit on `race_name`.
  - `build_models() -> dict[str, sklearn estimator]` — logistic regression (scaled), random forest, gradient boosting.
  - `evaluate(model, X_test, y_test) -> dict` — accuracy/precision/recall/f1/roc_auc + confusion values.
  - `train_global_model(input_path, target, output_dir) -> dict` — orchestrates; saves artifacts.
  - `main()` — CLI `--year --target`.
- Consumes: `model_ready_all_tracks.csv`, `FEATURE_COLUMNS`/`TARGET_COLUMN`/`LEAKAGE_COLUMNS` from Task 1.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model_training.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_global_model import race_aware_split, build_models, select_features


def _toy_model_ready(n_races=4, laps=40):
    rows = []
    for r in range(n_races):
        for l in range(1, laps + 1):
            rows.append({
                "year": 2025, "race_name": f"R{r}", "driver": "VER",
                "lap_number": l, "tyre_life": l % 20, "stint": 1 + l // 20,
                "compound": "SOFT" if l % 2 else "MEDIUM",
                "lap_percentage": l / laps, "total_laps": laps,
                "current_stint_lap": l % 20, "laps_since_pit": l % 20,
                "pit_stop_count_so_far": l // 20, "has_pitted": int(l > 20),
                "laps_since_last_pit": l % 20, "is_green_flag": 1,
                "is_safety_car": 0, "is_vsc": 0, "safety_car_or_vsc_flag": False,
                "position": 1 + (l % 5), "position_change": 0, "fresh_tyre": True,
                "round_number": r + 1, "track_name": f"R{r}", "session_type": "R",
                "team": "RB", "driver_number": 1,
                "will_pit_next_3_laps": int(l % 20 >= 17),
            })
    return pd.DataFrame(rows)


def test_race_aware_split_keeps_races_disjoint():
    df = _toy_model_ready()
    X, y, _ = select_features(df)
    train_idx, test_idx = race_aware_split(X, y, df["race_name"], test_size=0.5)
    train_races = set(df.iloc[train_idx]["race_name"])
    test_races = set(df.iloc[test_idx]["race_name"])
    assert train_races.isdisjoint(test_races)
    assert len(test_races) >= 1


def test_select_features_excludes_leakage():
    df = _toy_model_ready()
    X, y, names = select_features(df)
    for bad in ["pit_in_time", "pit_out_time", "pit_in_lap", "pit_out_lap",
                "will_pit_next_3_laps", "lap_time"]:
        assert bad not in names
    assert len(X) == len(y)


def test_build_models_returns_three_estimators():
    models = build_models()
    assert {"logistic_regression", "random_forest", "gradient_boosting"} <= set(models)
```

- [ ] **Step 2: Run to verify fail** → FAIL (import error).
- [ ] **Step 3: Implement `src/models/train_global_model.py`.** `select_features`: keep intersection of safe feature columns present in df, one-hot `compound` (and optionally `team`), coerce numerics, drop rows with NA target, cast bool→int. `race_aware_split`: `GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=...)` on groups; return first split. `build_models`: LR in a `Pipeline` with `StandardScaler`, `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`, `GradientBoostingClassifier(random_state=42)`. `evaluate`: sklearn metrics; `roc_auc` guarded by try/except + `predict_proba`. `train_global_model`: split, fit all models, pick best by f1, save best to `models/global/pit_strategy_model.pkl` (joblib), write `models/global/model_metadata.json` (features, target, best model name, split strategy, class balance, sklearn version), `outputs/reports/global_model_metrics.csv` (all models), `outputs/reports/race_level_model_metrics.csv` (best model metrics per test race), and figures `outputs/figures/confusion_matrix_global.png`, `outputs/figures/model_comparison.png` (matplotlib, `Agg` backend). `main()`: argparse `--year`, `--target` default `will_pit_next_3_laps`.
- [ ] **Step 4: Run tests** → PASS (3 tests).
- [ ] **Step 5: Run CLI** — `python src/models/train_global_model.py --year 2025 --target will_pit_next_3_laps` → trains, prints metrics, writes artifacts.
- [ ] **Step 6: Commit** — `git add src/models/train_global_model.py tests/test_model_training.py && git commit -m "feat: train global pit strategy model"`.

---

### Task 3 (Phase 7): Track comparison analysis

**Files:**
- Create: `src/analysis/__init__.py`, `src/analysis/track_comparison.py`

**Interfaces:**
- Produces:
  - `TRACK_GROUPS: dict[str, list[str]]` — analysis-only labels (street/high-deg/fast/mixed).
  - `pit_stops_by_race(df) -> pd.DataFrame`, `stint_length_by_race(df) -> pd.DataFrame`, `stint_length_by_compound_race(df) -> pd.DataFrame`, `compound_usage_by_race(df) -> pd.DataFrame`, `pit_lap_distribution(df) -> pd.DataFrame`, `sc_vsc_pit_impact(df) -> pd.DataFrame`.
  - `run_track_comparison(laps_path, fig_dir, report_path, race_metrics_path=None) -> None`
  - `main()` — CLI `--year`.
- Consumes: `laps_all_tracks.csv`; optionally `outputs/reports/race_level_model_metrics.csv` from Task 2.

- [ ] **Step 1:** Implement module (analysis aggregations + matplotlib `Agg` figures to `outputs/figures/track_comparison/`: `pit_stop_count_by_race.png`, `avg_stint_length_by_race.png`, `compound_usage_by_race.png`, `pit_lap_distribution_by_race.png`, and `race_level_model_performance.png` if race metrics exist). Write Markdown `outputs/reports/track_comparison_summary.md` summarizing tables + group labels. `TRACK_GROUPS` defined here only (not in pipeline). No TDD test required (figure-generation), but guard empty data.
- [ ] **Step 2: Run CLI** — `python src/analysis/track_comparison.py --year 2025` → figures + report written.
- [ ] **Step 3: Commit** — `git add src/analysis && git commit -m "feat: add track comparison analysis"`.

---

### Task 4 (Phase 8): Model explainability

**Files:**
- Create: `src/models/explain_model.py`

**Interfaces:**
- Produces:
  - `load_artifacts(model_path, data_path) -> tuple[model, pd.DataFrame]`
  - `global_feature_importance(model, feature_names) -> pd.DataFrame`
  - `permutation_feature_importance(model, X, y) -> pd.DataFrame`
  - `shap_summary(model, X, fig_path) -> bool` — returns False (and logs) if SHAP unsupported.
  - `run_explainability(model_path, data_path, fig_dir, report_path) -> None`
  - `main()` — CLI `--year`.
- Consumes: `models/global/pit_strategy_model.pkl`, `model_ready_all_tracks.csv`, feature list from metadata.

- [ ] **Step 1:** Implement. Reconstruct X via `select_features` (import from Task 2 for DRY). Feature importance from `.feature_importances_` or `abs(coef_)`. Permutation importance via `sklearn.inspection.permutation_importance` (n_repeats=5) on a sample (cap rows for speed). SHAP guarded in try/except (TreeExplainer for tree models on a sample); skip gracefully if it errors. Figures to `outputs/figures/explainability/`: `global_feature_importance.png`, `permutation_importance.png`, `shap_summary.png` (if produced). Write `outputs/reports/model_explainability_summary.md` answering the spec's questions (tire age vs track status, SC/VSC effect, Monaco difference, hardest tracks — derived from race-level metrics where available).
- [ ] **Step 2: Run CLI** — `python src/models/explain_model.py --year 2025` → figures + report.
- [ ] **Step 3: Commit** — `git add src/models/explain_model.py && git commit -m "feat: add model explainability"`.

---

### Task 5 (Phase 9): Upgrade Streamlit dashboard (all tracks)

**Files:**
- Create: `app/_shared.py`, `app/pages/1_Project_Overview.py`, `app/pages/2_Track_Explorer.py`, `app/pages/3_Strategy_Comparison.py`, `app/pages/4_Model_Predictions.py`, `app/pages/5_Explainability.py`
- Modify: `app/streamlit_app.py` (rewrite as multi-track entry; keep file path)

**Interfaces:**
- Consumes: `laps_all_tracks.csv`, `model_ready_all_tracks.csv`, `models/global/*`, reports & figures.
- Produces (in `app/_shared.py`):
  - `PROJECT_ROOT`, `load_laps()`, `load_model_ready()`, `load_model()`, `load_metadata()`, `safe_read_csv(path)`, `sidebar_filters(df) -> dict`, `apply_filters(df, filters) -> pd.DataFrame` — all `@st.cache_data` where appropriate, returning empty/None safely if files missing.

- [ ] **Step 1:** Implement `app/_shared.py` with cached loaders that never raise on missing files (return empty DataFrame / None + a flag). `streamlit_app.py` becomes the Overview landing using `_shared`. Build the 5 pages per spec, each guarded so a missing artifact shows an `st.info(...)` message instead of crashing. Sidebar filters: Year, Race, Driver, Team, Compound, Stint, Track status.
- [ ] **Step 2: Verify imports + data loading without launching server** — `python -c "import app._shared as s; print(len(s.load_laps()))"` and `python -m py_compile app/streamlit_app.py app/pages/*.py`. (Note: keep Monaco-specific old paths only as optional fallbacks; do not require them.)
- [ ] **Step 3: Commit** — `git add app && git commit -m "feat: update streamlit dashboard for all tracks"`.

---

### Task 6 (Phase 11): Testing & quality checks

**Files:**
- Modify: `tests/test_data_pipeline.py` (add data-quality checks), `tests/test_feature_engineering.py`, `tests/test_model_training.py`

**Interfaces:** Consumes all prior modules.

- [ ] **Step 1: Add data-quality tests** to `tests/test_data_pipeline.py` that run only if `model_ready_all_tracks.csv` exists (use `pytest.mark.skipif`/runtime skip): required columns present; no missing `year/race_name/driver/lap_number`; `lap_number > 0`; target has both classes; no dups on `[year,race_name,driver,lap_number]`; no inf values in numeric feature columns.

```python
# appended to tests/test_data_pipeline.py
import numpy as np
import pytest

MODEL_READY = PROJECT_ROOT / "data/processed/2025/model_ready_all_tracks.csv"


@pytest.mark.skipif(not MODEL_READY.exists(), reason="model-ready dataset not built")
def test_model_ready_quality():
    df = pd.read_csv(MODEL_READY)
    for col in ["year", "race_name", "driver", "lap_number", "will_pit_next_3_laps"]:
        assert col in df.columns
        assert df[col].notna().all()
    assert (df["lap_number"] > 0).all()
    assert set(df["will_pit_next_3_laps"].unique()) == {0, 1}
    assert not df.duplicated(subset=["year", "race_name", "driver", "lap_number"]).any()
    num = df.select_dtypes(include=[np.number])
    assert np.isfinite(num.to_numpy()).all()
```

- [ ] **Step 2: Run full suite** — `python -m pytest -q` → all PASS.
- [ ] **Step 3: Commit** — `git add tests && git commit -m "test: add pipeline and model quality checks"`.

---

### Task 7 (Phase 10): README & docs update

**Files:**
- Modify: `README.md`
- Create: `docs/project_workflow.md`, `docs/modeling_notes.md`, `docs/limitations.md`

**Interfaces:** documentation only.

- [ ] **Step 1:** Rewrite `README.md` with all spec sections (title, problem, why, Monaco→all-tracks evolution, data source = FastF1, workflow, repo structure, dataset structure, feature-engineering summary, modeling approach, **race-aware validation explanation**, key track-comparison findings, explainability summary, Streamlit instructions, how to run pipeline, how to run tests, limitations, future improvements). Write the three docs honestly (exploratory/educational; not professional strategy software). Pull real numbers from generated metrics/reports.
- [ ] **Step 2: Commit** — `git add README.md docs && git commit -m "docs: update readme for full-season project"`.

---

### Task 8 (Phase 12): Final verification

- [ ] **Step 1:** `python src/features/build_features.py --year 2025 --target-window 3`
- [ ] **Step 2:** `python src/models/train_global_model.py --year 2025 --target will_pit_next_3_laps`
- [ ] **Step 3:** `python -m pytest -q`
- [ ] **Step 4:** Verify Streamlit imports/data loading (no long-running server): `python -m py_compile app/streamlit_app.py app/pages/*.py` and import `app/_shared.py`.
- [ ] **Step 5:** Commit any regenerated artifacts: `git add data/processed/2025 outputs models && git commit -m "data: add model-ready dataset, metrics, figures, and trained model"`.

---

### Task 9: Finish branch

- [ ] Use superpowers:finishing-a-development-branch; verify tests; present integration options.

## Self-Review

- **Spec coverage:** Phase 5 → Task 1 (all suggested functions, target, CLI, outputs). Phase 6 → Task 2 (3 models, GroupShuffleSplit, metrics, artifacts, CLI). Phase 7 → Task 3 (all analyses, figures, report, track groups as labels). Phase 8 → Task 4 (importance/permutation/SHAP, questions, figures, report). Phase 9 → Task 5 (5 pages, filters, graceful degradation, command). Phase 10 → Task 7 (README + 3 docs). Phase 11 → Task 6 (quality checks, pytest). Phase 12 → Task 8 (verification commands + summary). ✓
- **Placeholder scan:** test code concrete; implementation steps name exact functions/paths/outputs. ✓
- **Type consistency:** `select_features` reused by Tasks 2 & 4; `FEATURE_COLUMNS`/`TARGET_COLUMN`/`LEAKAGE_COLUMNS` defined in Task 1 and consumed downstream; `TARGET_COLUMN = "will_pit_next_3_laps"` consistent. ✓
- **Note:** Phase 7/8 implemented as `src/` scripts (not notebooks) for CLI verifiability; the spec permits this when scripts suit the repo. Weather features documented as unavailable in model-ready set. `improve-codebase-architecture` skill unavailable — its principles applied manually.
