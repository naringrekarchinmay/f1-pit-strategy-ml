# Data Dictionary — Combined 2025 Multi-Track Lap Dataset

This dictionary documents the **standardized snake_case schema** produced by the
reusable loader (`src/data/load_fastf1_data.py`) and combined by the all-tracks
builder (`src/data/build_all_tracks_dataset.py`) into
`data/processed/2025/laps_all_tracks.csv`.

Conventions:
- **Type** is the logical type in the saved CSV (`int`, `float`, `string`, `bool`, `datetime`).
- **Required** = always present in the schema (filled with NA if the source lacks it). **Optional** = present only when the session provides it (weather block).
- **Origin**: `raw` = taken directly from a FastF1 field; `derived` = computed; `metadata` = added by the loader.
- **ML use** = likely usefulness as a future modeling feature.

---

## Core columns (required)

| Column | Meaning | Type | Required | Origin | ML use |
|---|---|---|---|---|---|
| `year` | Championship season (e.g. 2025) | int | Required | metadata | Yes (grouping) |
| `round_number` | Calendar round number of the event | int | Required | metadata (`session.event['RoundNumber']`) | Yes (track ordering) |
| `race_name` | Display race name (spec calendar name, e.g. `Monaco`) | string | Required | metadata | Yes (grouping) |
| `track_name` | Circuit / event name from FastF1 (`EventName`/`Location`) | string | Required | metadata | Yes (track identity) |
| `session_type` | Session code (`R`, `Q`, `FP1`, …) | string | Required | metadata | Yes (filter) |
| `driver` | Driver three-letter abbreviation (e.g. `VER`) | string | Required | raw (`Driver`) | Yes |
| `team` | Constructor / team name | string | Required | raw (`Team`) | Yes |
| `lap_number` | Lap index within the session | int | Required | raw (`LapNumber`) | Yes |

## Timing columns

| Column | Meaning | Type | Required | Origin | ML use |
|---|---|---|---|---|---|
| `lap_time` | Lap time as a string (HH:MM:SS.sss) | string | Required | raw (`LapTime`) | Indirect (use seconds) |
| `lap_time_seconds` | Lap time in seconds | float | Required | derived (`LapTime.total_seconds()`) | Yes (target) |
| `sector_1_time` | Sector 1 time in seconds | float | Required (NA if absent) | derived (`Sector1Time`) | Yes |
| `sector_2_time` | Sector 2 time in seconds | float | Required (NA if absent) | derived (`Sector2Time`) | Yes |
| `sector_3_time` | Sector 3 time in seconds | float | Required (NA if absent) | derived (`Sector3Time`) | Yes |

## Tire and stint columns

| Column | Meaning | Type | Required | Origin | ML use |
|---|---|---|---|---|---|
| `compound` | Tyre compound (`SOFT`/`MEDIUM`/`HARD`/`INTERMEDIATE`/`WET`) | string | Required | raw (`Compound`) | Yes |
| `tyre_life` | Laps completed on the current tyre set | float | Required | raw (`TyreLife`) | Yes (degradation) |
| `stint` | Stint number for the driver | float | Required | raw (`Stint`) | Yes |
| `fresh_tyre` | Whether the tyre set started fresh | bool | Required | raw (`FreshTyre`) | Yes |

## Pit-related columns

| Column | Meaning | Type | Required | Origin | ML use |
|---|---|---|---|---|---|
| `pit_in_time` | Timestamp the car entered the pit lane (string) | string | Required (NA if no pit) | raw (`PitInTime`) | Indirect |
| `pit_out_time` | Timestamp the car exited the pit lane (string) | string | Required (NA if no pit) | raw (`PitOutTime`) | Indirect |
| `pit_in_lap` | `lap_number` if this lap is a pit-in lap, else NA | float | Required (derived) | derived | Yes (strategy) |
| `pit_out_lap` | `lap_number` if this lap is a pit-out lap, else NA | float | Required (derived) | derived | Yes (strategy) |
| `is_pit_lap` | True if the lap involved a pit in or out | bool | Required (derived) | derived | Yes |
| `pit_stop_count` | Cumulative number of pit stops for the driver up to and including this lap | int | Required (derived) | derived | Yes (strategy) |

## Race condition columns

| Column | Meaning | Type | Required | Origin | ML use |
|---|---|---|---|---|---|
| `track_status` | FastF1 track-status code(s) for the lap (e.g. `1`=clear, `2`=yellow, `4`=SC, `6`/`7`=VSC) | string | Required | raw (`TrackStatus`) | Yes |
| `safety_car_or_vsc_flag` | True if `track_status` indicates SC (`4`) or VSC (`6`/`7`) | bool | Required (derived) | derived | Yes (pace context) |
| `position` | Track position at the end of the lap | float | Required | raw (`Position`) | Yes |

## Weather columns (optional — present only when session provides weather)

| Column | Meaning | Type | Required | Origin | ML use |
|---|---|---|---|---|---|
| `air_temp` | Air temperature (°C) | float | Optional | raw (`AirTemp`) | Yes |
| `track_temp` | Track surface temperature (°C) | float | Optional | raw (`TrackTemp`) | Yes |
| `humidity` | Relative humidity (%) | float | Optional | raw (`Humidity`) | Maybe |
| `pressure` | Air pressure (mbar) | float | Optional | raw (`Pressure`) | Maybe |
| `rainfall` | Whether rain was recorded | bool | Optional | raw (`Rainfall`) | Yes |
| `wind_speed` | Wind speed (m/s) | float | Optional | raw (`WindSpeed`) | Maybe |
| `wind_direction` | Wind direction (degrees) | float | Optional | raw (`WindDirection`) | Maybe |

> Weather is saved to a separate per-race file (`weather.csv`) keyed by time, because
> FastF1 weather samples are time-series and do not map one-to-one to laps. The lap
> dataset stays lap-grained; weather can be joined later during feature engineering.

## Metadata columns

| Column | Meaning | Type | Required | Origin | ML use |
|---|---|---|---|---|---|
| `source` | Data source identifier (`fastf1`) | string | Required | metadata | No (provenance) |
| `source_file` | Path of the per-race output CSV this row came from | string | Required | metadata | No (provenance) |
| `created_at` | UTC timestamp when the row was generated | datetime | Required | metadata | No (provenance) |
