"""Tests for the global pit-strategy model training module."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_global_model import race_aware_split, build_models, select_features


def _toy_model_ready(n_races=4, laps=40):
    rows = []
    for r in range(n_races):
        for l in range(1, laps + 1):
            rows.append(
                {
                    "year": 2025, "race_name": f"R{r}", "driver": "VER",
                    "lap_number": l, "tyre_life": l % 20, "stint": 1 + l // 20,
                    "compound": "SOFT" if l % 2 else "MEDIUM",
                    "lap_percentage": l / laps, "total_laps": laps,
                    "current_stint_lap": l % 20, "laps_since_pit": l % 20,
                    "pit_stop_count_so_far": l // 20, "has_pitted": int(l > 20),
                    "laps_since_last_pit": l % 20, "is_green_flag": 1,
                    "is_safety_car": 0, "is_vsc": 0, "safety_car_or_vsc_flag": 0,
                    "position": 1 + (l % 5), "position_change": 0, "fresh_tyre": 1,
                    "round_number": r + 1, "track_name": f"R{r}", "session_type": "R",
                    "team": "RB", "is_pit_lap": l % 20 == 0,
                    "will_pit_next_3_laps": int(l % 20 >= 17),
                }
            )
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
