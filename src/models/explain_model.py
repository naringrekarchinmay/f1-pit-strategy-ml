"""Explain the global pit-strategy model.

Computes built-in feature importance, permutation importance, and (if SHAP is
installed and compatible) a SHAP summary, then writes figures to
``outputs/figures/explainability/`` and a Markdown summary to
``outputs/reports/model_explainability_summary.md``.

CLI::

    python src/models/explain_model.py --year 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_global_model import load_model_ready, select_features  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "global"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "explainability"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("explain_model")


def load_artifacts(model_path: str | Path, data_path: str | Path):
    """Load the trained model and the model-ready dataset."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")
    model = joblib.load(model_path)
    df = load_model_ready(data_path)
    return model, df


def _final_estimator(model):
    """Return the underlying estimator (unwrap a sklearn Pipeline)."""
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model


def global_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Feature importance from tree importances or |coef|, NA-safe."""
    est = _final_estimator(model)
    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
        kind = "tree_importance"
    elif hasattr(est, "coef_"):
        importances = np.abs(np.ravel(est.coef_))
        kind = "abs_coefficient"
    else:
        logger.warning("Model exposes no native importances.")
        return pd.DataFrame(columns=["feature", "importance", "kind"])
    n = min(len(importances), len(feature_names))
    out = pd.DataFrame(
        {"feature": feature_names[:n], "importance": importances[:n], "kind": kind}
    )
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_feature_importance(model, X, y, sample: int = 4000) -> pd.DataFrame:
    """Permutation importance on a capped sample for speed."""
    if len(X) > sample:
        Xs = X.sample(sample, random_state=42)
        ys = y.loc[Xs.index]
    else:
        Xs, ys = X, y
    result = permutation_importance(model, Xs, ys, n_repeats=5, random_state=42, n_jobs=-1)
    out = pd.DataFrame(
        {"feature": list(X.columns), "importance": result.importances_mean,
         "std": result.importances_std}
    )
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def shap_summary(model, X, fig_path: Path, sample: int = 1500) -> bool:
    """Produce a SHAP summary plot. Returns False (and logs) if unsupported."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed; skipping SHAP summary.")
        return False
    try:
        est = _final_estimator(model)
        Xs = X.sample(min(sample, len(X)), random_state=42)
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(Xs)
        # Binary classifiers may return a list (per class) or 3D array.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        plt.figure()
        shap.summary_plot(shap_values, Xs, show=False)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary -> %s", fig_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP summary failed (%s); skipping.", exc)
        return False


def _barh(data: pd.DataFrame, title: str, path: Path, top: int = 15) -> None:
    d = data.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(d))))
    ax.barh(d["feature"], d["importance"], color="#9467bd")
    ax.set_xlabel("Importance"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run_explainability(
    model_path: str | Path,
    data_path: str | Path,
    fig_dir: str | Path = FIGURES_DIR,
    report_path: str | Path = REPORTS_DIR / "model_explainability_summary.md",
) -> None:
    """Run all explainability steps and write figures + report."""
    model, df = load_artifacts(model_path, data_path)
    X, y, feature_names = select_features(df)

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fi = global_feature_importance(model, feature_names)
    if not fi.empty:
        _barh(fi, "Global feature importance", fig_dir / "global_feature_importance.png")

    pi = permutation_feature_importance(model, X, y)
    _barh(pi, "Permutation importance", fig_dir / "permutation_importance.png")

    shap_ok = shap_summary(model, X, fig_dir / "shap_summary.png")

    # Race-level difficulty from existing metrics (for the "hardest tracks" Q).
    race_metrics_path = REPORTS_DIR / "race_level_model_metrics.csv"
    race_metrics = pd.read_csv(race_metrics_path) if race_metrics_path.is_file() else None

    _write_report(report_path, fi, pi, shap_ok, race_metrics)
    logger.info("Explainability complete: figures -> %s", fig_dir)


def _write_report(report_path, fi, pi, shap_ok, race_metrics) -> None:
    top_feats = pi.head(8)["feature"].tolist() if not pi.empty else []

    def _has(substr):
        return any(substr in f for f in top_feats)

    lines = ["# Model Explainability Summary", ""]
    lines.append("This summarizes what drives the global `will_pit_next_3_laps` model. "
                 "Predictions are **exploratory and educational**, not professional race "
                 "strategy advice.")
    lines.append("")

    if not fi.empty:
        lines.append("## Top features (native importance)")
        lines.append(fi.head(10).to_markdown(index=False))
        lines.append("")
    if not pi.empty:
        lines.append("## Top features (permutation importance, held-out aware)")
        lines.append(pi.head(10).to_markdown(index=False))
        lines.append("")

    lines.append("## Answers to key questions")
    lines.append(f"- **Which features most influence pit prediction?** "
                 f"Top permutation features: {', '.join(top_feats[:5]) or 'n/a'}.")
    tire_terms = _has("tyre_life") or _has("laps_since_pit") or _has("current_stint_lap") or _has("stint")
    status_terms = _has("safety_car") or _has("is_vsc") or _has("track_status") or _has("is_green")
    lines.append(f"- **Tire age vs track status:** tyre/stint features "
                 f"{'rank highly' if tire_terms else 'appear modest'}; track-status flags "
                 f"{'also contribute' if status_terms else 'contribute less'}.")
    lines.append(f"- **Do SC/VSC periods matter?** Safety-car/VSC flags "
                 f"{'appear among influential features' if status_terms else 'are weaker predictors here'}.")
    if race_metrics is not None and not race_metrics.empty:
        rm = race_metrics.sort_values("f1")
        hardest = rm.head(3)["race_name"].tolist()
        easiest = rm.tail(3)["race_name"].tolist()
        monaco_row = race_metrics[race_metrics["race_name"] == "Monaco"]
        lines.append(f"- **Hardest test races (lowest F1):** {', '.join(hardest)}.")
        lines.append(f"- **Best-predicted test races (highest F1):** {', '.join(easiest)}.")
        if not monaco_row.empty:
            lines.append(f"- **Does Monaco behave differently?** Monaco F1 = "
                         f"{monaco_row['f1'].iloc[0]:.3f} on held-out evaluation "
                         f"(present in the test split).")
        else:
            lines.append("- **Does Monaco behave differently?** Monaco was in the training "
                         "split this run, so no held-out Monaco metric is available.")
    else:
        lines.append("- **Race difficulty / Monaco:** race-level metrics not found; run "
                     "training first.")
    lines.append("")

    lines.append("## Figures")
    lines.append("- `outputs/figures/explainability/global_feature_importance.png`")
    lines.append("- `outputs/figures/explainability/permutation_importance.png`")
    if shap_ok:
        lines.append("- `outputs/figures/explainability/shap_summary.png`")
    else:
        lines.append("- SHAP summary skipped (SHAP unavailable or incompatible with this model).")
    lines.append("")

    Path(report_path).write_text("\n".join(lines))
    logger.info("Wrote report -> %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain the global pit-strategy model.")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    data_path = PROCESSED_DIR / str(args.year) / "model_ready_all_tracks.csv"
    run_explainability(MODEL_DIR / "pit_strategy_model.pkl", data_path)
    print(REPORTS_DIR / "model_explainability_summary.md")


if __name__ == "__main__":
    main()
