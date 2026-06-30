# Limitations

This project is an **exploratory, educational** analysis of Formula 1 pit
strategy. It is **not** professional race-strategy software, and its predictions
should not be used to make real strategic decisions.

## Modeling limitations

- **Modest predictive performance.** The best baseline (logistic regression)
  reaches ROC-AUC ≈ 0.70 and F1 ≈ 0.24 for `will_pit_next_3_laps` on **held-out
  races**. Recall is moderate while precision is low — the model flags many pit
  windows it is unsure about. This is honest baseline behaviour, not a solved
  problem.
- **Class imbalance.** Only ~8.5% of driver-laps are positive (a pit within the
  next 3 laps). Accuracy is therefore a misleading metric; we report precision,
  recall, F1 and ROC-AUC alongside it.
- **Race-aware validation lowers scores on purpose.** We hold entire races out
  of training (GroupShuffleSplit on `race_name`). Random row splitting would
  produce higher but misleading numbers because laps from the same race are
  highly correlated.

## Data limitations

- **Event-name resolution.** FastF1 display-name matching is ambiguous (it once
  mapped "Great Britain" to the Austrian GP). Races are now resolved to an exact
  calendar round, but new seasons may need the alias table updated.
- **Weather is not yet a feature.** Weather is collected per race
  (`weather.csv`) but is a separate time series; it is **not** joined into the
  model-ready dataset, so weather features are currently unused.
- **Pit detection depends on FastF1 timing.** Pit-in/out are inferred from
  FastF1 fields, which occasionally miss laps or mark laps inaccurate.
- **Single season.** Only 2025 is included; there is no cross-season validation.

## Scope

- No claim of strategic optimality, causal inference, or real-time readiness.
- Track groupings (street / high-degradation / fast / mixed) are **analysis
  labels only** and are not used by the pipeline or model.
