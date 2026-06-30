# Multi-Track F1 Data Pipeline (Phase 1–4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the Monaco-only F1 pit-strategy prototype into a reusable, year/race-organized FastF1 data pipeline covering all 2025 races, without breaking existing Monaco work.

**Architecture:** Refactor the Monaco data-collection notebook logic into a parameterized module (`src/data/load_fastf1_data.py`) that loads any FastF1 race session, standardizes it to a snake_case schema, and writes per-race CSVs under `data/processed/<year>/<race>/`. A second script (`src/data/build_all_tracks_dataset.py`) drives that loader across the full 2025 calendar, tolerating individual race failures, and combines successes into one dataset plus a build-summary report.

**Tech Stack:** Python 3.13, FastF1 3.8, pandas 2.2, pytest 8.3, argparse, logging.

## Global Constraints

- Complete **only Phase 1–4**. Do NOT train models, add explainability, or redesign the dashboard. Do NOT proceed to Phase 5 without explicit approval.
- Preserve all existing Monaco files; do not delete or rewrite them.
- No hard-coded Monaco-specific logic in reusable modules.
- Outputs organized by `year` then `race` (snake_case race folder).
- snake_case column names in all standardized outputs.
- Clear logging via `logging`; never silently fail — log a warning/error when data is missing.
- A failing race must not stop the all-races build.
- Save intermediate CSVs.
- Standardized lap schema must contain these columns (filled with NA when source lacks them): `year, round_number, race_name, track_name, session_type, driver, team, lap_number, lap_time, lap_time_seconds, sector_1_time, sector_2_time, sector_3_time, compound, tyre_life, stint, fresh_tyre, pit_in_time, pit_out_time, pit_in_lap, pit_out_lap, is_pit_lap, pit_stop_count, track_status, safety_car_or_vsc_flag, position, source, source_file, created_at`.

---

## File Structure

- `docs/phase_1_repo_audit.md` — Phase 1 audit (docs only).
- `docs/data_dictionary.md` — Phase 2 column dictionary.
- `data/raw/2025/`, `data/processed/2025/`, `outputs/figures/`, `outputs/reports/`, `outputs/predictions/`, `src/data/`, `src/features/`, `src/models/`, `tests/` — Phase 2 directories (`.gitkeep` only where empty and worth tracking).
- `src/__init__.py`, `src/data/__init__.py` — package markers.
- `src/data/load_fastf1_data.py` — Phase 3 reusable loader + CLI.
- `src/data/build_all_tracks_dataset.py` — Phase 4 all-races builder + CLI.
- `tests/test_data_pipeline.py` — lightweight pipeline tests.

---

### Task 1: Phase 1 — Repository audit document

**Files:**
- Create: `docs/phase_1_repo_audit.md`

**Interfaces:**
- Produces: documentation only (no code symbols).

- [ ] **Step 1:** Write `docs/phase_1_repo_audit.md` covering: current repo structure summary; Monaco-specific files (notebooks 01–05, `data/raw/2025_monaco_*.csv`, `data/processed/2025_monaco_*.csv`, `models/monaco_2025_*.pkl`, `outputs/metrics/monaco_2025_*`, `outputs/figures/monaco_2025_*`, `app/streamlit_app.py`); hard-coded values needing refactor (`RACE_NAME="Monaco"`, `YEAR=2025`, `../data/...` relative paths in notebooks, `2025_monaco_*` filename prefixes); current data files; model/output files; dashboard files; recommended changes for Phases 2–4; risks before expanding to all tracks (FastF1 event-name mismatches, rate limiting / long load times, missing weather/sector fields per race, cache size, schema drift).
- [ ] **Step 2:** Commit.

```bash
git add docs/phase_1_repo_audit.md
git commit -m "docs: add Phase 1 repository audit"
```

---

### Task 2: Phase 2 — Multi-track folder structure + data dictionary

**Files:**
- Create: `data/raw/2025/.gitkeep`, `data/processed/2025/.gitkeep`, `outputs/reports/.gitkeep`, `outputs/predictions/.gitkeep`, `src/__init__.py`, `src/data/__init__.py`, `src/features/.gitkeep`, `src/models/.gitkeep`, `tests/.gitkeep`
- Create: `docs/data_dictionary.md`

**Interfaces:**
- Produces: directory layout consumed by Tasks 3–4; data dictionary doc.

- [ ] **Step 1:** Create the directories and `.gitkeep`/`__init__.py` marker files listed above (do not move or delete existing Monaco files).
- [ ] **Step 2:** Write `docs/data_dictionary.md` documenting every column in the Global Constraints schema plus optional weather columns (`air_temp, track_temp, humidity, pressure, rainfall, wind_speed, wind_direction`). For each column give: meaning, data type, required/optional, raw/derived, ML usefulness.
- [ ] **Step 3:** Commit.

```bash
git add data/raw/2025/.gitkeep data/processed/2025/.gitkeep outputs/reports/.gitkeep outputs/predictions/.gitkeep src/__init__.py src/data/__init__.py src/features/.gitkeep src/models/.gitkeep tests/.gitkeep docs/data_dictionary.md
git commit -m "feat: add multi-track folder structure and data dictionary"
```

---

### Task 3: Phase 3 — Reusable FastF1 loader

**Files:**
- Create: `src/data/load_fastf1_data.py`
- Test: `tests/test_data_pipeline.py` (created here, extended in Task 5)

**Interfaces:**
- Produces:
  - `normalize_race_name(race_name: str) -> str` — ASCII snake_case slug (e.g. `"Saudi Arabia"`→`"saudi_arabia"`, `"São Paulo"`→`"sao_paulo"`).
  - `setup_fastf1_cache(cache_dir: Path | None = None) -> Path`
  - `load_race_session(year: int, race_name: str, session_type: str = "R")` → loaded FastF1 `Session`.
  - `extract_lap_data(session, year, race_name, session_type) -> pd.DataFrame` — standardized schema (Global Constraints column list).
  - `extract_weather_data(session, year, race_name, session_type) -> pd.DataFrame`
  - `save_race_outputs(laps_df, weather_df, year, race_name) -> dict` — returns `{"laps_path", "weather_path"}`.
  - `STANDARD_LAP_COLUMNS: list[str]` — the canonical ordered column list.
  - `main()` — argparse CLI: `--year --race --session`.
- Consumes: directories from Task 2.

- [ ] **Step 1: Write failing tests for `normalize_race_name` and schema**

```python
# tests/test_data_pipeline.py
import pandas as pd
from src.data.load_fastf1_data import normalize_race_name, STANDARD_LAP_COLUMNS


def test_normalize_race_name_basic():
    assert normalize_race_name("Monaco") == "monaco"
    assert normalize_race_name("Saudi Arabia") == "saudi_arabia"
    assert normalize_race_name("  Great Britain ") == "great_britain"


def test_normalize_race_name_unicode():
    assert normalize_race_name("São Paulo") == "sao_paulo"


def test_standard_columns_cover_required_fields():
    required = {
        "year", "round_number", "race_name", "track_name", "session_type",
        "driver", "team", "lap_number", "lap_time", "compound", "tyre_life",
        "stint", "track_status", "is_pit_lap", "pit_stop_count",
    }
    assert required.issubset(set(STANDARD_LAP_COLUMNS))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_data_pipeline.py -v`
Expected: FAIL (ModuleNotFoundError / ImportError — module not yet created).

- [ ] **Step 3: Implement `src/data/load_fastf1_data.py`**

Implement with: `PROJECT_ROOT` derived from `__file__` (so paths work from any cwd); `logging` configured; `normalize_race_name` using `unicodedata` NFKD ASCII fold + regex to snake_case; `setup_fastf1_cache` enabling FastF1 cache at `data/cache` (mkdir first); `load_race_session` calling `fastf1.get_session(...).load()` with try/except + logging; `extract_lap_data` mapping FastF1 columns to `STANDARD_LAP_COLUMNS`, converting timedeltas to seconds (`lap_time_seconds`, `sector_*_time` as seconds, keep `lap_time` as string), deriving `is_pit_lap` (PitIn/Out not null), `pit_stop_count` (cumulative pit-ins per driver), `pit_in_lap`/`pit_out_lap`, `safety_car_or_vsc_flag` (TrackStatus contains 4/6/7), and metadata (`round_number`, `track_name` from `session.event`; `source="fastf1"`, `source_file`, `created_at`); missing source columns filled with `pd.NA`; `extract_weather_data` standardizing weather columns (empty DataFrame if none); `save_race_outputs` writing `data/processed/<year>/<race>/race_laps.csv` and `weather.csv`; `main()` argparse CLI. Every function has a docstring; guard optional fields with `if col in df`.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_data_pipeline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify CLI against real Monaco data**

Run: `python src/data/load_fastf1_data.py --year 2025 --race Monaco --session R`
Expected: logs progress; writes `data/processed/2025/monaco/race_laps.csv` and `weather.csv`; no crash.

- [ ] **Step 6: Commit**

```bash
git add src/data/load_fastf1_data.py tests/test_data_pipeline.py
git commit -m "feat: add reusable FastF1 race data loader"
```

---

### Task 4: Phase 4 — All-races dataset builder

**Files:**
- Create: `src/data/build_all_tracks_dataset.py`

**Interfaces:**
- Produces:
  - `RACES_2025: list[str]` — the 24-race calendar (display names from the spec).
  - `build_all_tracks(year: int, session_type: str = "R", races: list[str] | None = None) -> pd.DataFrame` — returns the build summary DataFrame.
  - `combine_race_files(summary_df, year) -> pd.DataFrame` — concatenates successful per-race laps.
  - `main()` — argparse CLI `--year --session`.
- Consumes: `load_race_session`, `extract_lap_data`, `extract_weather_data`, `save_race_outputs`, `normalize_race_name`, `setup_fastf1_cache` from Task 3.

- [ ] **Step 1: Write failing test for failure tolerance + combine logic**

```python
# appended to tests/test_data_pipeline.py
def test_combine_skips_failed_races(tmp_path, monkeypatch):
    import src.data.build_all_tracks_dataset as b
    summary = pd.DataFrame([
        {"year": 2025, "race_name": "Monaco", "session_type": "R",
         "status": "success", "rows_loaded": 2, "output_path": "ok.csv",
         "error_message": ""},
        {"year": 2025, "race_name": "Nowhere", "session_type": "R",
         "status": "failed", "rows_loaded": 0, "output_path": "",
         "error_message": "boom"},
    ])
    good = tmp_path / "ok.csv"
    pd.DataFrame({"year": [2025, 2025], "race_name": ["Monaco", "Monaco"],
                  "driver": ["VER", "VER"], "lap_number": [1, 2]}).to_csv(good, index=False)
    summary.loc[0, "output_path"] = str(good)
    combined = b.combine_race_files(summary, 2025)
    assert len(combined) == 2
    assert set(combined["race_name"]) == {"Monaco"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_data_pipeline.py::test_combine_skips_failed_races -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/data/build_all_tracks_dataset.py`**

`RACES_2025` lists the 24 races from the spec. `build_all_tracks` loops races, wrapping each in try/except: load session, extract laps+weather, save per-race outputs, append a summary row (`status`, `rows_loaded`, `output_path`, `error_message`); a failure logs and continues. `combine_race_files` reads each successful `output_path`, concatenates, deduplicates on `["year","race_name","driver","lap_number"]` where present, writes `data/processed/<year>/laps_all_tracks.csv`. `main()` writes summary to `outputs/reports/all_tracks_build_summary.csv` with columns `year, race_name, session_type, status, rows_loaded, output_path, error_message`.

- [ ] **Step 4: Run test to verify pass**

Run: `python -m pytest tests/test_data_pipeline.py::test_combine_skips_failed_races -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/build_all_tracks_dataset.py tests/test_data_pipeline.py
git commit -m "feat: add all-tracks 2025 dataset builder"
```

---

### Task 5: Tests + full verification

**Files:**
- Modify: `tests/test_data_pipeline.py`

**Interfaces:**
- Consumes: all symbols from Tasks 3–4.

- [ ] **Step 1: Add remaining required tests** — required-columns-present-after-extraction (using a small synthetic FastF1-like DataFrame passed through the standardization path or asserting `STANDARD_LAP_COLUMNS` on a built frame), missing-field safety (extraction fills NA when a source column is absent), and duplicate-row check on `year/race_name/driver/lap_number`.
- [ ] **Step 2: Run full suite**

Run: `python -m pytest -v`
Expected: all PASS.

- [ ] **Step 3: Verify Monaco loader CLI** (already in Task 3 Step 5; re-run to confirm idempotent).

Run: `python src/data/load_fastf1_data.py --year 2025 --race Monaco --session R`

- [ ] **Step 4: Run all-tracks builder** (best-effort; record failures, do not block).

Run: `python src/data/build_all_tracks_dataset.py --year 2025 --session R`
Expected: writes `data/processed/2025/laps_all_tracks.csv` + `outputs/reports/all_tracks_build_summary.csv`.

- [ ] **Step 5: Commit any test additions, then finishing-a-development-branch.**

```bash
git add tests/test_data_pipeline.py
git commit -m "test: complete data pipeline test coverage"
```

---

## Self-Review

- **Spec coverage:** Phase 1 (Task 1), Phase 2 structure + data dictionary (Task 2), Phase 3 loader + all suggested functions + CLI (Task 3), Phase 4 builder + race list + summary columns (Task 4), all 5 minimum tests + verification commands (Tasks 3–5). ✓
- **Placeholders:** Test steps include concrete code; implementation steps specify exact symbols, columns, and paths. ✓
- **Type consistency:** Function names/signatures in the Interfaces blocks match across Tasks 3–4 (`normalize_race_name`, `extract_lap_data`, `combine_race_files`, `STANDARD_LAP_COLUMNS`). ✓
