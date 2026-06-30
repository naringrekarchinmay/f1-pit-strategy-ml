# Track Comparison Summary (2025)

- Races analyzed: **24**
- Total laps: **26,689**
- Drivers: **21**

> Track groupings are analysis labels only and are not used by the data pipeline or model.

## Average pit stops per driver (top 5)
| race_name   |   avg_pit_stops |
|:------------|----------------:|
| Australia   |         4.82353 |
| Canada      |         4.15    |
| Spain       |         2.89474 |
| Qatar       |         2.31579 |
| Bahrain     |         2.15    |

## Average stint length by race (top 5 longest)
| race_name     |   avg_stint_length |
|:--------------|-------------------:|
| Singapore     |            28.5814 |
| Hungary       |            27.9184 |
| Mexico City   |            26.8723 |
| United States |            26.0244 |
| Japan         |            25.8293 |

## Pit stops under safety car / VSC (top 5 share)
| race_name      |   pit_under_sc_vsc_share |
|:---------------|-------------------------:|
| Australia      |                 0.865854 |
| Emilia Romagna |                 0.648649 |
| Canada         |                 0.638554 |
| Netherlands    |                 0.6      |
| Miami          |                 0.421053 |

## Track groupings (analysis labels)
- **street**: Monaco, Singapore, Azerbaijan, Las Vegas, Saudi Arabia
- **high_degradation**: Bahrain, Spain, Hungary, Qatar
- **fast_flowing**: Great Britain, Belgium, Italy, Japan
- **mixed**: Canada, Austria, Mexico City, São Paulo, Abu Dhabi

## Race-level model performance (held-out test races)
Model performance is reported only for races held out of training (race-aware split), so coverage is a subset of all races.

| model               | race_name      |   n_laps |   n_positives |   accuracy |   precision |    recall |       f1 |
|:--------------------|:---------------|---------:|--------------:|-----------:|------------:|----------:|---------:|
| logistic_regression | Great Britain  |      825 |           106 |   0.780606 |   0.337662  | 0.735849  | 0.462908 |
| logistic_regression | Abu Dhabi      |     1156 |            81 |   0.916955 |   0.368421  | 0.259259  | 0.304348 |
| logistic_regression | Emilia Romagna |     1207 |           107 |   0.700083 |   0.173913  | 0.635514  | 0.273092 |
| logistic_regression | Italy          |      974 |            60 |   0.650924 |   0.14467   | 0.95      | 0.251101 |
| logistic_regression | Monaco         |     1425 |           110 |   0.496842 |   0.0859482 | 0.572727  | 0.149466 |
| logistic_regression | Qatar          |     1067 |           128 |   0.854733 |   0.2       | 0.0703125 | 0.104046 |

## Figures
- `outputs/figures/track_comparison/pit_stop_count_by_race.png`
- `outputs/figures/track_comparison/avg_stint_length_by_race.png`
- `outputs/figures/track_comparison/compound_usage_by_race.png`
- `outputs/figures/track_comparison/pit_lap_distribution_by_race.png`
- `outputs/figures/track_comparison/race_level_model_performance.png`
