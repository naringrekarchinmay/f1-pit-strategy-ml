"""Tests for the multi-track feature engineering layer."""
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (used in later quality checks / fixtures)
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    create_pit_window_target,
    add_track_status_features,
    add_pit_history_features,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
)


def _toy():
    # One driver, one race, 6 laps, pits (pit_in) on lap 4.
    return pd.DataFrame(
        {
            "year": [2025] * 6,
            "race_name": ["Test"] * 6,
            "driver": ["VER"] * 6,
            "lap_number": [1, 2, 3, 4, 5, 6],
            "stint": [1, 1, 1, 1, 2, 2],
            "tyre_life": [1, 2, 3, 4, 1, 2],
            "pit_in_time": [pd.NA, pd.NA, pd.NA, "0:55:00", pd.NA, pd.NA],
            "is_pit_lap": [False, False, False, True, False, False],
            "track_status": ["1", "1", "4", "1", "67", "1"],
        }
    )


def test_target_marks_laps_before_pit():
    out = create_pit_window_target(_toy(), window_laps=3)
    # Pit-in is lap 4 -> laps 1,2,3 are within 3 laps before -> 1.
    assert out.loc[out["lap_number"] == 1, TARGET_COLUMN].iloc[0] == 1
    assert out.loc[out["lap_number"] == 3, TARGET_COLUMN].iloc[0] == 1
    # Laps 5,6 have no upcoming pit -> 0.
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
