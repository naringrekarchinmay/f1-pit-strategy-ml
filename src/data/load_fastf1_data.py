"""Reusable FastF1 race-data loader.

Loads any FastF1 race session, standardizes the lap and weather data into a
fixed snake_case schema, and writes per-race CSV files organized by year and
race:

    data/processed/<year>/<race>/race_laps.csv
    data/processed/<year>/<race>/weather.csv

The module is parameterized by ``(year, race_name, session_type)`` and contains
no Monaco-specific logic. It can be used as a library (import the functions) or
from the command line::

    python src/data/load_fastf1_data.py --year 2025 --race Monaco --session R
"""
from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:  # FastF1 is optional at import time so unit tests can run without it.
    import fastf1
except ImportError:  # pragma: no cover - exercised only when fastf1 is absent
    fastf1 = None


# --------------------------------------------------------------------------- #
# Paths & logging
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("load_fastf1_data")


# --------------------------------------------------------------------------- #
# Standardized schema
# --------------------------------------------------------------------------- #
STANDARD_LAP_COLUMNS: list[str] = [
    # core
    "year",
    "round_number",
    "race_name",
    "track_name",
    "session_type",
    "driver",
    "team",
    "lap_number",
    # timing
    "lap_time",
    "lap_time_seconds",
    "sector_1_time",
    "sector_2_time",
    "sector_3_time",
    # tire & stint
    "compound",
    "tyre_life",
    "stint",
    "fresh_tyre",
    # pit
    "pit_in_time",
    "pit_out_time",
    "pit_in_lap",
    "pit_out_lap",
    "is_pit_lap",
    "pit_stop_count",
    # race condition
    "track_status",
    "safety_car_or_vsc_flag",
    "position",
    # metadata
    "source",
    "source_file",
    "created_at",
]

STANDARD_WEATHER_COLUMNS: list[str] = [
    "year",
    "round_number",
    "race_name",
    "track_name",
    "session_type",
    "time",
    "air_temp",
    "track_temp",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "source",
    "created_at",
]

# FastF1 track-status codes that indicate a Safety Car (4) or Virtual Safety
# Car deployed/ending (6/7).
_SC_VSC_CODES = {"4", "6", "7"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_race_name(race_name: str) -> str:
    """Return an ASCII snake_case slug for a race name.

    Used for output folder names so paths are filesystem-safe and consistent
    across operating systems.

    >>> normalize_race_name("Saudi Arabia")
    'saudi_arabia'
    >>> normalize_race_name("São Paulo")
    'sao_paulo'
    """
    # Fold accented characters to their ASCII base (São -> Sao).
    ascii_name = (
        unicodedata.normalize("NFKD", str(race_name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    ascii_name = ascii_name.strip().lower()
    # Replace any run of non-alphanumeric characters with a single underscore.
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name)
    return slug.strip("_")


def _timedelta_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a timedelta-like Series to float seconds, NA-safe."""
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def _safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return ``df[name]`` if present, otherwise an all-NA Series.

    Keeps extraction robust when a race/session lacks an expected field.
    """
    if name in df.columns:
        return df[name]
    logger.warning("Source column '%s' missing; filling with NA.", name)
    return pd.Series([pd.NA] * len(df), index=df.index)


def setup_fastf1_cache(cache_dir: Path | None = None) -> Path:
    """Enable the FastF1 on-disk cache and return its path."""
    cache_path = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)
    if fastf1 is None:
        raise ImportError("fastf1 is not installed; cannot enable cache.")
    fastf1.Cache.enable_cache(str(cache_path))
    logger.info("FastF1 cache enabled at %s", cache_path)
    return cache_path


# --------------------------------------------------------------------------- #
# Event resolution
# --------------------------------------------------------------------------- #
# Map common display names to the exact FastF1 ``EventName``. This avoids
# FastF1's fuzzy matching, which can resolve an ambiguous display name to the
# wrong round (e.g. "Great Britain" -> the Austrian GP, or "United States" ->
# Miami/Las Vegas, which all share the "United States" country).
EVENT_NAME_ALIASES: dict[str, str] = {
    "australia": "Australian Grand Prix",
    "china": "Chinese Grand Prix",
    "japan": "Japanese Grand Prix",
    "bahrain": "Bahrain Grand Prix",
    "saudi arabia": "Saudi Arabian Grand Prix",
    "miami": "Miami Grand Prix",
    "emilia romagna": "Emilia Romagna Grand Prix",
    "monaco": "Monaco Grand Prix",
    "spain": "Spanish Grand Prix",
    "canada": "Canadian Grand Prix",
    "austria": "Austrian Grand Prix",
    "great britain": "British Grand Prix",
    "britain": "British Grand Prix",
    "belgium": "Belgian Grand Prix",
    "hungary": "Hungarian Grand Prix",
    "netherlands": "Dutch Grand Prix",
    "italy": "Italian Grand Prix",
    "azerbaijan": "Azerbaijan Grand Prix",
    "singapore": "Singapore Grand Prix",
    "united states": "United States Grand Prix",
    "mexico city": "Mexico City Grand Prix",
    "mexico": "Mexico City Grand Prix",
    "sao paulo": "São Paulo Grand Prix",
    "são paulo": "São Paulo Grand Prix",
    "las vegas": "Las Vegas Grand Prix",
    "qatar": "Qatar Grand Prix",
    "abu dhabi": "Abu Dhabi Grand Prix",
}


def resolve_event_round(year: int, race_name: str) -> int | None:
    """Resolve a display race name to its calendar round number.

    Uses an explicit ``EventName`` alias table first (exact, unambiguous), then
    falls back to a case-insensitive substring search over EventName / Country /
    Location. Returns ``None`` if no confident match is found, in which case the
    caller may fall back to FastF1's own name matching.
    """
    if fastf1 is None:
        return None
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load %d event schedule: %s", year, exc)
        return None

    key = str(race_name).strip().lower()
    alias = EVENT_NAME_ALIASES.get(key)
    if alias is not None:
        exact = schedule[schedule["EventName"] == alias]
        if not exact.empty:
            return int(exact["RoundNumber"].iloc[0])

    # Fallback: substring match across name/country/location (skip testing=round 0).
    for col in ("EventName", "Country", "Location"):
        if col not in schedule.columns:
            continue
        hits = schedule[
            schedule[col].astype(str).str.lower().str.contains(key, na=False)
            & (schedule["RoundNumber"] > 0)
        ]
        if len(hits) == 1:
            return int(hits["RoundNumber"].iloc[0])
    logger.warning("Could not confidently resolve round for '%s' %d.", race_name, year)
    return None


# --------------------------------------------------------------------------- #
# Loading & extraction
# --------------------------------------------------------------------------- #
def load_race_session(year: int, race_name: str, session_type: str = "R"):
    """Load and return a FastF1 session for the given race.

    Resolves the race to an exact calendar round first (avoiding ambiguous
    fuzzy matches), then loads by round number. Falls back to FastF1's name
    matching only if the round cannot be resolved.
    """
    if fastf1 is None:
        raise ImportError("fastf1 is not installed; cannot load sessions.")
    logger.info("Loading session: %s %s (%s)", year, race_name, session_type)
    round_number = resolve_event_round(year, race_name)
    try:
        if round_number is not None:
            logger.info("Resolved '%s' -> %d round %d", race_name, year, round_number)
            session = fastf1.get_session(year, round_number, session_type)
        else:
            logger.warning("Falling back to FastF1 name matching for '%s'.", race_name)
            session = fastf1.get_session(year, race_name, session_type)
        session.load()
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        logger.error("Failed to load %s %s (%s): %s", year, race_name, session_type, exc)
        raise
    logger.info("Loaded %s %s (%s)", year, race_name, session_type)
    return session


def _event_metadata(session, year: int, race_name: str) -> dict:
    """Extract round number and track name from a session's event, NA-safe."""
    round_number = pd.NA
    track_name = pd.NA
    event = getattr(session, "event", None)
    if event is not None:
        try:
            round_number = int(event["RoundNumber"])
        except (KeyError, TypeError, ValueError):
            logger.warning("RoundNumber unavailable for %s.", race_name)
        for key in ("EventName", "OfficialEventName", "Location"):
            try:
                value = event[key]
            except (KeyError, TypeError):
                continue
            if isinstance(value, str) and value:
                track_name = value
                break
    return {"round_number": round_number, "track_name": track_name}


def extract_lap_data(session, year: int, race_name: str, session_type: str) -> pd.DataFrame:
    """Standardize a session's laps into :data:`STANDARD_LAP_COLUMNS`.

    Missing source fields are filled with NA and logged; nothing fails silently.
    """
    laps = session.laps.copy() if getattr(session, "laps", None) is not None else pd.DataFrame()
    if laps.empty:
        logger.warning("No lap data for %s %s; returning empty frame.", year, race_name)
        return pd.DataFrame(columns=STANDARD_LAP_COLUMNS)

    meta = _event_metadata(session, year, race_name)
    created_at = datetime.now(timezone.utc).isoformat()

    pit_in = _safe_column(laps, "PitInTime")
    pit_out = _safe_column(laps, "PitOutTime")
    lap_number = pd.to_numeric(_safe_column(laps, "LapNumber"), errors="coerce")
    is_pit_lap = pit_in.notna() | pit_out.notna()
    track_status = _safe_column(laps, "TrackStatus").astype("string")

    out = pd.DataFrame(index=laps.index)
    out["year"] = year
    out["round_number"] = meta["round_number"]
    out["race_name"] = race_name
    out["track_name"] = meta["track_name"]
    out["session_type"] = session_type
    out["driver"] = _safe_column(laps, "Driver").astype("string")
    out["team"] = _safe_column(laps, "Team").astype("string")
    out["lap_number"] = lap_number

    out["lap_time"] = _safe_column(laps, "LapTime").astype("string")
    out["lap_time_seconds"] = _timedelta_to_seconds(_safe_column(laps, "LapTime"))
    out["sector_1_time"] = _timedelta_to_seconds(_safe_column(laps, "Sector1Time"))
    out["sector_2_time"] = _timedelta_to_seconds(_safe_column(laps, "Sector2Time"))
    out["sector_3_time"] = _timedelta_to_seconds(_safe_column(laps, "Sector3Time"))

    out["compound"] = _safe_column(laps, "Compound").astype("string")
    out["tyre_life"] = pd.to_numeric(_safe_column(laps, "TyreLife"), errors="coerce")
    out["stint"] = pd.to_numeric(_safe_column(laps, "Stint"), errors="coerce")
    out["fresh_tyre"] = _safe_column(laps, "FreshTyre")

    out["pit_in_time"] = pit_in.astype("string")
    out["pit_out_time"] = pit_out.astype("string")
    out["pit_in_lap"] = lap_number.where(pit_in.notna())
    out["pit_out_lap"] = lap_number.where(pit_out.notna())
    out["is_pit_lap"] = is_pit_lap

    # Cumulative pit-stop count per driver (count pit-in events up to this lap).
    out["pit_stop_count"] = (
        pit_in.notna().groupby(out["driver"]).cumsum().astype("Int64")
    )

    out["track_status"] = track_status
    out["safety_car_or_vsc_flag"] = track_status.fillna("").apply(
        lambda s: any(code in str(s) for code in _SC_VSC_CODES)
    )
    out["position"] = pd.to_numeric(_safe_column(laps, "Position"), errors="coerce")

    out["source"] = "fastf1"
    out["source_file"] = pd.NA  # filled in save_race_outputs once the path is known
    out["created_at"] = created_at

    # Guarantee column order/presence.
    out = out.reindex(columns=STANDARD_LAP_COLUMNS)
    logger.info("Extracted %d laps for %s %s.", len(out), year, race_name)
    return out.reset_index(drop=True)


def extract_weather_data(session, year: int, race_name: str, session_type: str) -> pd.DataFrame:
    """Standardize a session's weather data; empty frame if none available."""
    weather = (
        session.weather_data.copy()
        if getattr(session, "weather_data", None) is not None
        else pd.DataFrame()
    )
    if weather.empty:
        logger.warning("No weather data for %s %s.", year, race_name)
        return pd.DataFrame(columns=STANDARD_WEATHER_COLUMNS)

    meta = _event_metadata(session, year, race_name)
    out = pd.DataFrame(index=weather.index)
    out["year"] = year
    out["round_number"] = meta["round_number"]
    out["race_name"] = race_name
    out["track_name"] = meta["track_name"]
    out["session_type"] = session_type
    out["time"] = _safe_column(weather, "Time").astype("string")
    out["air_temp"] = pd.to_numeric(_safe_column(weather, "AirTemp"), errors="coerce")
    out["track_temp"] = pd.to_numeric(_safe_column(weather, "TrackTemp"), errors="coerce")
    out["humidity"] = pd.to_numeric(_safe_column(weather, "Humidity"), errors="coerce")
    out["pressure"] = pd.to_numeric(_safe_column(weather, "Pressure"), errors="coerce")
    out["rainfall"] = _safe_column(weather, "Rainfall")
    out["wind_speed"] = pd.to_numeric(_safe_column(weather, "WindSpeed"), errors="coerce")
    out["wind_direction"] = pd.to_numeric(
        _safe_column(weather, "WindDirection"), errors="coerce"
    )
    out["source"] = "fastf1"
    out["created_at"] = datetime.now(timezone.utc).isoformat()

    out = out.reindex(columns=STANDARD_WEATHER_COLUMNS)
    logger.info("Extracted %d weather samples for %s %s.", len(out), year, race_name)
    return out.reset_index(drop=True)


def save_race_outputs(
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    year: int,
    race_name: str,
) -> dict:
    """Write per-race lap and weather CSVs and return their paths.

    Output layout: ``data/processed/<year>/<race_slug>/race_laps.csv`` and
    ``weather.csv``.
    """
    race_slug = normalize_race_name(race_name)
    out_dir = PROCESSED_DIR / str(year) / race_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    laps_path = out_dir / "race_laps.csv"
    weather_path = out_dir / "weather.csv"

    # Record provenance now that we know the destination path.
    if not laps_df.empty and "source_file" in laps_df.columns:
        laps_df = laps_df.copy()
        laps_df["source_file"] = str(laps_path)

    laps_df.to_csv(laps_path, index=False)
    weather_df.to_csv(weather_path, index=False)
    logger.info("Saved laps -> %s (%d rows)", laps_path, len(laps_df))
    logger.info("Saved weather -> %s (%d rows)", weather_path, len(weather_df))
    return {"laps_path": str(laps_path), "weather_path": str(weather_path)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """Command-line entry point: load one race and save standardized CSVs."""
    parser = argparse.ArgumentParser(
        description="Load and standardize a FastF1 race session into CSVs."
    )
    parser.add_argument("--year", type=int, required=True, help="Season year, e.g. 2025")
    parser.add_argument("--race", type=str, required=True, help="Race name, e.g. Monaco")
    parser.add_argument(
        "--session", type=str, default="R", help="Session type (default: R)"
    )
    args = parser.parse_args()

    setup_fastf1_cache()
    session = load_race_session(args.year, args.race, args.session)
    laps_df = extract_lap_data(session, args.year, args.race, args.session)
    weather_df = extract_weather_data(session, args.year, args.race, args.session)
    paths = save_race_outputs(laps_df, weather_df, args.year, args.race)

    logger.info(
        "Done: %s %s (%s) -> %d laps, %d weather rows.",
        args.year,
        args.race,
        args.session,
        len(laps_df),
        len(weather_df),
    )
    print(paths["laps_path"])
    print(paths["weather_path"])


if __name__ == "__main__":
    main()
