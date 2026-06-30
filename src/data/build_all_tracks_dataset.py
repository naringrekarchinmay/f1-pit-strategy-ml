"""Build a combined 2025 multi-track lap dataset from FastF1.

Loops over the full 2025 race calendar, calls the reusable loader in
:mod:`src.data.load_fastf1_data` for each race, saves per-race CSVs, then
combines all successfully loaded races into a single dataset and writes a build
summary report.

A failure in one race is logged and recorded in the summary; it never aborts
the rest of the build.

CLI::

    python src/data/build_all_tracks_dataset.py --year 2025 --session R

Outputs:
    data/processed/<year>/laps_all_tracks.csv
    outputs/reports/all_tracks_build_summary.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make the project importable when run as a script (python src/data/...).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_fastf1_data import (  # noqa: E402
    PROCESSED_DIR,
    extract_lap_data,
    extract_weather_data,
    load_race_session,
    save_race_outputs,
    setup_fastf1_cache,
)

logger = logging.getLogger("build_all_tracks_dataset")

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# 2025 F1 calendar (display names). FastF1's get_session resolves these to its
# own event names; mismatches are handled per-race via try/except.
RACES_2025: list[str] = [
    "Australia",
    "China",
    "Japan",
    "Bahrain",
    "Saudi Arabia",
    "Miami",
    "Emilia Romagna",
    "Monaco",
    "Spain",
    "Canada",
    "Austria",
    "Great Britain",
    "Belgium",
    "Hungary",
    "Netherlands",
    "Italy",
    "Azerbaijan",
    "Singapore",
    "United States",
    "Mexico City",
    "São Paulo",
    "Las Vegas",
    "Qatar",
    "Abu Dhabi",
]

SUMMARY_COLUMNS: list[str] = [
    "year",
    "race_name",
    "session_type",
    "status",
    "rows_loaded",
    "output_path",
    "error_message",
]

# Deduplicate combined laps on these keys when all are present.
DEDUP_KEYS = ["year", "race_name", "driver", "lap_number"]


def build_all_tracks(
    year: int,
    session_type: str = "R",
    races: list[str] | None = None,
) -> pd.DataFrame:
    """Load every race in ``races`` and return a build-summary DataFrame.

    Each race is wrapped in try/except so one failure does not stop the build.
    Per-race CSVs are written via :func:`save_race_outputs`.
    """
    race_list = races if races is not None else RACES_2025
    setup_fastf1_cache()

    rows: list[dict] = []
    for race_name in race_list:
        logger.info("=== Building %s %s (%s) ===", year, race_name, session_type)
        try:
            session = load_race_session(year, race_name, session_type)
            laps_df = extract_lap_data(session, year, race_name, session_type)
            weather_df = extract_weather_data(session, year, race_name, session_type)
            paths = save_race_outputs(laps_df, weather_df, year, race_name)
            rows.append(
                {
                    "year": year,
                    "race_name": race_name,
                    "session_type": session_type,
                    "status": "success" if len(laps_df) else "empty",
                    "rows_loaded": len(laps_df),
                    "output_path": paths["laps_path"],
                    "error_message": "",
                }
            )
            logger.info("OK %s: %d laps.", race_name, len(laps_df))
        except Exception as exc:  # noqa: BLE001 - tolerate per-race failures
            logger.error("FAILED %s %s: %s", year, race_name, exc)
            rows.append(
                {
                    "year": year,
                    "race_name": race_name,
                    "session_type": session_type,
                    "status": "failed",
                    "rows_loaded": 0,
                    "output_path": "",
                    "error_message": str(exc),
                }
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def combine_race_files(summary_df: pd.DataFrame, year: int, write: bool = True) -> pd.DataFrame:
    """Concatenate per-race lap CSVs for successful races into one DataFrame.

    Reads each successful row's ``output_path``, skips missing/failed races,
    deduplicates on :data:`DEDUP_KEYS` where present, and (when ``write``) saves
    ``data/processed/<year>/laps_all_tracks.csv``.
    """
    frames: list[pd.DataFrame] = []
    successful = summary_df[summary_df["status"] == "success"]
    for _, row in successful.iterrows():
        path = Path(str(row["output_path"]))
        if not path.is_file():
            logger.warning("Skipping %s: file not found at %s", row["race_name"], path)
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: could not read %s (%s)", row["race_name"], path, exc)

    if not frames:
        logger.warning("No race files to combine.")
        combined = pd.DataFrame()
    else:
        combined = pd.concat(frames, ignore_index=True)
        dedup_keys = [k for k in DEDUP_KEYS if k in combined.columns]
        if dedup_keys:
            before = len(combined)
            combined = combined.drop_duplicates(subset=dedup_keys).reset_index(drop=True)
            logger.info("Deduplicated combined laps: %d -> %d rows.", before, len(combined))

    if write:
        out_dir = PROCESSED_DIR / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "laps_all_tracks.csv"
        combined.to_csv(out_path, index=False)
        logger.info("Saved combined dataset -> %s (%d rows)", out_path, len(combined))

    return combined


def main() -> None:
    """CLI entry point: build all races, combine, and write the summary report."""
    parser = argparse.ArgumentParser(
        description="Build the combined 2025 multi-track lap dataset."
    )
    parser.add_argument("--year", type=int, default=2025, help="Season year (default: 2025)")
    parser.add_argument(
        "--session", type=str, default="R", help="Session type (default: R)"
    )
    args = parser.parse_args()

    summary_df = build_all_tracks(args.year, args.session)
    combined = combine_race_files(summary_df, args.year, write=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "all_tracks_build_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    n_ok = int((summary_df["status"] == "success").sum())
    n_failed = int((summary_df["status"] == "failed").sum())
    logger.info(
        "Build complete: %d/%d races succeeded, %d failed, %d combined laps.",
        n_ok,
        len(summary_df),
        n_failed,
        len(combined),
    )
    logger.info("Summary report -> %s", summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()
