# Model Explainability Summary

This summarizes what drives the global `will_pit_next_3_laps` model. Predictions are **exploratory and educational**, not professional race strategy advice.

## Top features (native importance)
| feature               |   importance | kind            |
|:----------------------|-------------:|:----------------|
| lap_percentage        |     2.74756  | abs_coefficient |
| lap_number            |     2.49991  | abs_coefficient |
| stint                 |     1.46541  | abs_coefficient |
| pit_stop_count_so_far |     1.33363  | abs_coefficient |
| laps_since_last_pit   |     0.39713  | abs_coefficient |
| total_laps            |     0.38825  | abs_coefficient |
| round_number          |     0.379276 | abs_coefficient |
| has_pitted            |     0.374714 | abs_coefficient |
| compound_HARD         |     0.306482 | abs_coefficient |
| laps_since_pit        |     0.287858 | abs_coefficient |

## Top features (permutation importance, held-out aware)
| feature                |   importance |         std |
|:-----------------------|-------------:|------------:|
| lap_percentage         |      0.14295 | 0.006495    |
| lap_number             |      0.12235 | 0.0031289   |
| stint                  |      0.0758  | 0.00296395  |
| pit_stop_count_so_far  |      0.05855 | 0.0024052   |
| total_laps             |      0.0248  | 0.000827647 |
| round_number           |      0.0079  | 0.00221698  |
| safety_car_or_vsc_flag |      0.0042  | 0.000927362 |
| is_vsc                 |      0.0041  | 0.00140178  |
| compound_MEDIUM        |      0.0032  | 0.00175642  |
| is_green_flag          |      0.00195 | 0.00160779  |

## Answers to key questions
- **Which features most influence pit prediction?** Top permutation features: lap_percentage, lap_number, stint, pit_stop_count_so_far, total_laps.
- **Tire age vs track status:** tyre/stint features rank highly; track-status flags also contribute.
- **Do SC/VSC periods matter?** Safety-car/VSC flags appear among influential features.
- **Hardest test races (lowest F1):** Qatar, Monaco, Italy.
- **Best-predicted test races (highest F1):** Emilia Romagna, Abu Dhabi, Great Britain.
- **Does Monaco behave differently?** Monaco F1 = 0.149 on held-out evaluation (present in the test split).

## Figures
- `outputs/figures/explainability/global_feature_importance.png`
- `outputs/figures/explainability/permutation_importance.png`
- `outputs/figures/explainability/shap_summary.png`
