# Modeling Notes

## Target

`will_pit_next_3_laps` — for each driver-lap, **1** if that driver makes a pit
stop within the next 3 laps, else **0**. Pit events are detected from FastF1
`pit_in_time`. The current lap is excluded from the look-ahead, so the target is
never trivially self-predicting.

Class balance (2025, all tracks): ~**8.5% positive** (2,271 of 26,689 laps).

## Leakage avoidance

Features use **current-and-past** lap information only. The following columns are
explicitly excluded from features (`LEAKAGE_COLUMNS` in
`src/features/build_features.py`): `lap_time`, `pit_in_time`, `pit_out_time`,
`pit_in_lap`, `pit_out_lap`, `source`, `source_file`, `created_at`.

`pit_stop_count_so_far` counts pit-ins *strictly before* the current lap, so it
cannot reveal whether the current lap is a pit lap.

A data bug was caught via this lens: FastF1 had resolved "Great Britain" to the
Austrian GP, duplicating a race. With one copy in train and one in test the
model scored a perfect held-out F1 — a classic leakage signature. Fixed by
resolving races to exact calendar rounds.

## Features (leakage-safe)

- **Race context:** `round_number`, `lap_number`, `total_laps`, `lap_percentage`
- **Position:** `position`, `position_change`
- **Tyre / stint:** `compound`, `tyre_life`, `stint`, `fresh_tyre`,
  `laps_since_pit`, `current_stint_lap`
- **Pit history (past only):** `is_pit_lap`, `pit_stop_count_so_far`,
  `has_pitted`, `laps_since_last_pit`
- **Track condition:** `is_green_flag`, `is_safety_car`, `is_vsc`,
  `safety_car_or_vsc_flag`

`compound` is one-hot encoded; numeric features are passed through (logistic
regression is additionally standard-scaled in a pipeline).

## Validation

**Race-aware** via `GroupShuffleSplit(test_size=0.25)` on `race_name` — whole
races are held out. This is the only validation used for headline metrics; plain
random row splitting is deliberately avoided.

## Baselines & results (held-out races)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.72 | 0.16 | 0.50 | **0.24** | 0.70 |
| Random Forest | 0.90 | 0.25 | 0.07 | 0.11 | 0.69 |
| Gradient Boosting | 0.89 | 0.22 | 0.07 | 0.11 | 0.72 |

Best model by F1: **logistic regression** (saved to
`models/global/pit_strategy_model.pkl`). High tree-model accuracy is mostly the
majority class; F1/recall show the real difficulty.

## What drives predictions

Permutation importance (held-out aware) ranks **race progress** features highest:
`lap_percentage`, `lap_number`, `stint`, `pit_stop_count_so_far`, `total_laps`.
Safety-car/VSC flags contribute weakly. In other words, *when* in the race a lap
occurs and how far into a stint a driver is matter most; track-status flags are
secondary for this target.
