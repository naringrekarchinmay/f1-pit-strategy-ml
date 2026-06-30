"""Lightweight tests for the multi-track F1 data pipeline.

Run with: ``python -m pytest`` from the project root.
"""
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is importable when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_fastf1_data import (
    normalize_race_name,
    extract_lap_data,
    STANDARD_LAP_COLUMNS,
)


class _FakeEvent(dict):
    """Minimal stand-in for a FastF1 event row (subscriptable)."""


class _FakeSession:
    """Minimal FastF1 session stub exposing ``laps`` and ``event``."""

    def __init__(self, laps: pd.DataFrame):
        self.laps = laps
        self.event = _FakeEvent({"RoundNumber": 8, "EventName": "Test Grand Prix"})
        self.weather_data = pd.DataFrame()


def _sample_laps(include_optional: bool = True) -> pd.DataFrame:
    """Build a small FastF1-style laps frame for extraction tests."""
    data = {
        "Driver": ["VER", "VER", "HAM"],
        "Team": ["Red Bull", "Red Bull", "Ferrari"],
        "LapNumber": [1, 2, 1],
        "LapTime": pd.to_timedelta(["0:01:27", "0:01:46", "0:01:30"]),
        "Compound": ["HARD", "HARD", "MEDIUM"],
        "TyreLife": [1, 2, 1],
        "Stint": [1, 1, 1],
        "FreshTyre": [True, True, True],
        "PitInTime": [pd.NaT, pd.to_timedelta("0:59:00"), pd.NaT],
        "PitOutTime": [pd.NaT, pd.NaT, pd.NaT],
        "TrackStatus": ["1", "4", "1"],
        "Position": [1, 1, 3],
    }
    if include_optional:
        data["Sector1Time"] = pd.to_timedelta(["0:00:30", "0:00:31", "0:00:32"])
    return pd.DataFrame(data)


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


def test_combine_skips_failed_races(tmp_path):
    import src.data.build_all_tracks_dataset as b

    good = tmp_path / "ok.csv"
    pd.DataFrame(
        {
            "year": [2025, 2025],
            "race_name": ["Monaco", "Monaco"],
            "driver": ["VER", "VER"],
            "lap_number": [1, 2],
        }
    ).to_csv(good, index=False)

    summary = pd.DataFrame(
        [
            {
                "year": 2025, "race_name": "Monaco", "session_type": "R",
                "status": "success", "rows_loaded": 2, "output_path": str(good),
                "error_message": "",
            },
            {
                "year": 2025, "race_name": "Nowhere", "session_type": "R",
                "status": "failed", "rows_loaded": 0, "output_path": "",
                "error_message": "boom",
            },
        ]
    )

    combined = b.combine_race_files(summary, 2025, write=False)
    assert len(combined) == 2
    assert set(combined["race_name"]) == {"Monaco"}


def test_extract_lap_data_has_required_columns():
    session = _FakeSession(_sample_laps())
    out = extract_lap_data(session, 2025, "Testland", "R")
    # Standardized schema present and in order.
    assert list(out.columns) == STANDARD_LAP_COLUMNS
    assert len(out) == 3
    # Metadata + derived fields populated.
    assert (out["year"] == 2025).all()
    assert (out["race_name"] == "Testland").all()
    assert out["round_number"].iloc[0] == 8
    assert out["lap_time_seconds"].iloc[0] == 87.0
    # Derived pit + flag logic.
    assert bool(out["is_pit_lap"].iloc[1]) is True
    assert bool(out["safety_car_or_vsc_flag"].iloc[1]) is True
    # Cumulative pit count is per-driver.
    assert out["pit_stop_count"].tolist() == [0, 1, 0]


def test_extract_lap_data_missing_fields_safe():
    # Drop optional sector columns entirely; extraction must fill NA, not crash.
    session = _FakeSession(_sample_laps(include_optional=False))
    out = extract_lap_data(session, 2025, "Testland", "R")
    assert list(out.columns) == STANDARD_LAP_COLUMNS
    assert out["sector_1_time"].isna().all()


def test_no_duplicate_laps_per_driver():
    session = _FakeSession(_sample_laps())
    out = extract_lap_data(session, 2025, "Testland", "R")
    keys = ["year", "race_name", "driver", "lap_number"]
    assert not out.duplicated(subset=keys).any()
