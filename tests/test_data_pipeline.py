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
