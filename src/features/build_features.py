"""Multi-track feature engineering for the F1 pit-strategy model.

Converts the Phase-4 all-track lap dataset
(``data/processed/<year>/laps_all_tracks.csv``) into a leakage-safe,
model-ready dataset with a binary target ``will_pit_next_3_laps``.

Design notes
------------
* Every feature is computed from **current-or-past** lap information only.
  Columns that reveal the future or the target directly are listed in
  :data:`LEAKAGE_COLUMNS` and are never used as model features.
* Weather columns are *not* present in ``laps_all_tracks.csv`` (FastF1 weather
  is a separate time series saved per race), so they are not included here; the
  dataset is honestly "model-ready" only for the lap-grained features below.

CLI::

    python src/features/build_features.py --year 2025 --target-window 3
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_features")

TARGET_COLUMN = "will_pit_next_3_laps"

# Columns that must never be used as model features (future / target leakage,
# free-text, or provenance).
LEAKAGE_COLUMNS = [
    "lap_time",
    "pit_in_time",
    "pit_out_time",
    "pit_in_lap",
    "pit_out_lap",
    "source",
    "source_file",
    "created_at",
]

# Safe, leakage-free features for modeling (subset actually present is used).
FEATURE_COLUMNS = [
    # race context
    "round_number",
    "lap_number",
    "total_laps",
    "lap_percentage",
    # driver / team / position
    "position",
    "position_change",
    # tire & stint
    "tyre_life",
    "stint",
    "fresh_tyre",
    "laps_since_pit",
    "current_stint_lap",
    # pit history (computed from PAST laps only)
    "is_pit_lap",
    "pit_stop_count_so_far",
    "has_pitted",
    "laps_since_last_pit",
    # track condition
    "is_green_flag",
    "is_safety_car",
    "is_vsc",
    "safety_car_or_vsc_flag",
    # categorical (encoded downstream)
    "compound",
]

_GROUP_KEYS = ["year", "race_name", "driver"]


# --------------------------------------------------------------------------- #
# Loading & standardization
# --------------------------------------------------------------------------- #
def load_laps_dataset(path: str | Path) -> pd.DataFrame:
    """Load the all-track laps CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Laps dataset not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded laps dataset %s (%d rows, %d cols)", path, len(df), df.shape[1])
    return df


def standardize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce key columns to consistent dtypes and sort for windowed features."""
    df = df.copy()
    for col in ["year", "round_number", "lap_number", "stint", "tyre_life", "position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "track_status" in df.columns:
        df["track_status"] = df["track_status"].astype("string")
    if "fresh_tyre" in df.columns:
        df["fresh_tyre"] = df["fresh_tyre"].astype("boolean").fillna(False).astype(int)
    sort_cols = [c for c in ["year", "race_name", "driver", "lap_number"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
def add_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``total_laps`` and ``lap_percentage`` per race."""
    df = df.copy()
    df["total_laps"] = df.groupby(["year", "race_name"])["lap_number"].transform("max")
    df["lap_percentage"] = (df["lap_number"] / df["total_laps"]).clip(upper=1.0)
    return df


def add_tire_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``current_stint_lap`` and ``laps_since_pit`` (within-stint age)."""
    df = df.copy()
    grp = df.groupby(_GROUP_KEYS + ["stint"])
    df["current_stint_lap"] = grp.cumcount() + 1
    df["laps_since_pit"] = df["current_stint_lap"] - 1
    return df


def add_pit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pit-history features computed from PAST laps only (no leakage).

    ``pit_stop_count_so_far`` counts pit-ins strictly *before* the current lap,
    so it never reveals whether the current lap is a pit lap.
    """
    df = df.copy()
    pit_event = df["pit_in_time"].notna() if "pit_in_time" in df.columns else pd.Series(
        [False] * len(df), index=df.index
    )
    df["_pit_event"] = pit_event.astype(int)

    cum = df.groupby(_GROUP_KEYS)["_pit_event"].cumsum()
    # Count of pits BEFORE this lap = cumulative minus this lap's own event.
    df["pit_stop_count_so_far"] = (cum - df["_pit_event"]).astype(int)
    df["has_pitted"] = (df["pit_stop_count_so_far"] > 0).astype(int)

    # laps_since_last_pit: laps elapsed since the most recent PRIOR pit-in lap.
    # Vectorized: mark pit laps, shift by one (so the current lap's own pit does
    # not count), forward-fill within group, then subtract.
    df["_pit_lap_marker"] = df["lap_number"].where(df["_pit_event"] == 1)
    shifted = df.groupby(_GROUP_KEYS)["_pit_lap_marker"].shift()
    last_pit = shifted.groupby([df[k] for k in _GROUP_KEYS]).ffill()
    df["laps_since_last_pit"] = (
        (df["lap_number"] - last_pit).where(last_pit.notna(), df["lap_number"]).astype(int)
    )
    df = df.drop(columns=["_pit_event", "_pit_lap_marker"])
    return df


def add_track_status_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decode FastF1 ``track_status`` codes into binary flags.

    Codes may be concatenated per lap (e.g. ``'124'``). Code meanings:
    1=clear, 2=yellow, 4=safety car, 5=red, 6/7=virtual safety car.
    """
    df = df.copy()
    status = df.get("track_status", pd.Series([""] * len(df), index=df.index))
    status = status.fillna("").astype(str)
    df["is_safety_car"] = status.str.contains("4").astype(int)
    df["is_vsc"] = (status.str.contains("6") | status.str.contains("7")).astype(int)
    # Green = only clear running: contains '1' and none of the incident codes.
    incident = status.str.contains("[24567]", regex=True)
    df["is_green_flag"] = ((~incident) & status.str.contains("1")).astype(int)
    if "safety_car_or_vsc_flag" in df.columns:
        df["safety_car_or_vsc_flag"] = (
            df["safety_car_or_vsc_flag"].astype("boolean").fillna(False).astype(int)
        )
    else:
        df["safety_car_or_vsc_flag"] = (
            (df["is_safety_car"] == 1) | (df["is_vsc"] == 1)
        ).astype(int)
    return df


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``position_change`` (delta vs previous lap; +ve means lost places)."""
    df = df.copy()
    if "position" in df.columns:
        df["position_change"] = (
            df.groupby(_GROUP_KEYS)["position"].diff().fillna(0)
        )
    else:
        df["position_change"] = 0
    return df


def create_pit_window_target(df: pd.DataFrame, window_laps: int = 3) -> pd.DataFrame:
    """Add :data:`TARGET_COLUMN` = 1 if a pit-in occurs in the next N laps.

    Looks ahead within each (year, race, driver) group only. The current lap is
    NOT counted, avoiding trivial self-prediction.
    """
    df = df.copy()
    pit_event = df["pit_in_time"].notna() if "pit_in_time" in df.columns else pd.Series(
        [False] * len(df), index=df.index
    )
    df["_pit_event"] = pit_event.astype(int)

    target = pd.Series(0, index=df.index, dtype=int)
    for _, group in df.groupby(_GROUP_KEYS):
        pit_laps = set(group.loc[group["_pit_event"] == 1, "lap_number"].tolist())
        if not pit_laps:
            continue
        for idx, lap in group["lap_number"].items():
            upcoming = range(int(lap) + 1, int(lap) + window_laps + 1)
            if any(p in pit_laps for p in upcoming):
                target.at[idx] = 1
    df[TARGET_COLUMN] = target
    df = df.drop(columns=["_pit_event"])
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table describing each column's role and missingness."""
    rows = []
    for col in df.columns:
        if col == TARGET_COLUMN:
            role = "target"
        elif col in FEATURE_COLUMNS:
            role = "feature"
        elif col in LEAKAGE_COLUMNS:
            role = "excluded_leakage"
        else:
            role = "identifier_or_raw"
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "role": role,
                "n_missing": int(df[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_model_ready_dataset(
    input_path: str | Path,
    output_path: str | Path,
    target_window: int = 3,
) -> pd.DataFrame:
    """Run the full feature pipeline and save the model-ready CSV + summary."""
    df = load_laps_dataset(input_path)
    df = standardize_feature_columns(df)
    df = add_race_context_features(df)
    df = add_tire_stint_features(df)
    df = add_pit_history_features(df)
    df = add_track_status_features(df)
    df = add_position_features(df)
    df = create_pit_window_target(df, window_laps=target_window)

    df["feature_built_at"] = datetime.now(timezone.utc).isoformat()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved model-ready dataset -> %s (%d rows)", output_path, len(df))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "feature_engineering_summary.csv"
    _feature_summary(df).to_csv(summary_path, index=False)
    logger.info("Saved feature summary -> %s", summary_path)

    balance = df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    logger.info(
        "Target '%s' balance: %s (%d positives / %d rows)",
        TARGET_COLUMN,
        {k: round(v, 4) for k, v in balance.items()},
        int(df[TARGET_COLUMN].sum()),
        len(df),
    )
    n_features = len([c for c in FEATURE_COLUMNS if c in df.columns])
    logger.info("Usable model features present: %d", n_features)
    return df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the model-ready feature dataset.")
    parser.add_argument("--year", type=int, default=2025, help="Season year (default 2025)")
    parser.add_argument(
        "--target-window", type=int, default=3, help="Pit look-ahead window in laps"
    )
    args = parser.parse_args()

    input_path = PROCESSED_DIR / str(args.year) / "laps_all_tracks.csv"
    output_path = PROCESSED_DIR / str(args.year) / "model_ready_all_tracks.csv"
    build_model_ready_dataset(input_path, output_path, target_window=args.target_window)
    print(output_path)


if __name__ == "__main__":
    main()
