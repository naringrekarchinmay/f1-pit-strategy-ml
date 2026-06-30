"""Train a baseline global pit-strategy model across all 2025 tracks.

Predicts the binary target ``will_pit_next_3_laps`` using only leakage-safe
features. Validation is **race-aware**: whole races are held out of training via
``GroupShuffleSplit`` on ``race_name`` (never a plain random row split), because
random splitting would leak within-race correlations and overstate performance.

CLI::

    python src/models/train_global_model.py --year 2025 --target will_pit_next_3_laps
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless figure rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (  # noqa: E402
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "global"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_global_model")

# Categorical features that are one-hot encoded.
CATEGORICAL_FEATURES = ["compound"]


# --------------------------------------------------------------------------- #
# Data prep
# --------------------------------------------------------------------------- #
def load_model_ready(path: str | Path) -> pd.DataFrame:
    """Load the model-ready dataset produced by feature engineering."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model-ready dataset not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded model-ready dataset %s (%d rows)", path, len(df))
    return df


def select_features(df: pd.DataFrame):
    """Return ``(X, y, feature_names)`` using only safe, leakage-free features.

    Numeric features are kept as-is; ``compound`` is one-hot encoded. Rows with a
    missing target are dropped. Returns a numeric feature matrix ready for sklearn.
    """
    df = df.copy()
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not in dataset.")
    df = df[df[TARGET_COLUMN].notna()].copy()
    y = df[TARGET_COLUMN].astype(int)

    numeric_feats = [
        c
        for c in FEATURE_COLUMNS
        if c in df.columns and c not in CATEGORICAL_FEATURES and c not in LEAKAGE_COLUMNS
    ]
    X = df[numeric_feats].copy()
    # Coerce booleans / numerics; fill residual NA with 0 (e.g. position).
    for col in numeric_feats:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    # One-hot encode present categorical features.
    for cat in CATEGORICAL_FEATURES:
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat].astype("string").fillna("UNKNOWN"), prefix=cat)
            X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    feature_names = list(X.columns)
    return X.reset_index(drop=True), y.reset_index(drop=True), feature_names


def race_aware_split(X, y, groups, test_size: float = 0.25, random_state: int = 42):
    """Split indices keeping whole races (groups) disjoint between train/test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return train_idx, test_idx


def build_models() -> dict:
    """Return the baseline model zoo (simple, interpretable first)."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test) -> dict:
    """Compute classification metrics; roc_auc guarded for predict_proba."""
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    try:
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, proba)
    except Exception as exc:  # noqa: BLE001
        logger.warning("roc_auc unavailable: %s", exc)
        metrics["roc_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def _plot_confusion(metrics: dict, path: Path) -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No pit", "Pit ≤3"]); ax.set_yticklabels(["No pit", "Pit ≤3"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (global, held-out races)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_model_comparison(all_metrics: dict, path: Path) -> None:
    names = list(all_metrics)
    f1s = [all_metrics[n]["f1"] for n in names]
    aucs = [all_metrics[n].get("roc_auc", float("nan")) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, f1s, width=0.4, label="F1")
    ax.bar(x + 0.2, aucs, width=0.4, label="ROC AUC")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Model comparison (held-out races)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _race_level_metrics(model, X_test, y_test, race_test) -> pd.DataFrame:
    rows = []
    df = pd.DataFrame({"race_name": race_test.values, "y": y_test.values})
    df["pred"] = model.predict(X_test)
    for race, g in df.groupby("race_name"):
        rows.append(
            {
                "race_name": race,
                "n_laps": len(g),
                "n_positives": int(g["y"].sum()),
                "accuracy": accuracy_score(g["y"], g["pred"]),
                "precision": precision_score(g["y"], g["pred"], zero_division=0),
                "recall": recall_score(g["y"], g["pred"], zero_division=0),
                "f1": f1_score(g["y"], g["pred"], zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_global_model(input_path: str | Path, target: str, output_dir: str | Path) -> dict:
    """Train all baseline models with race-aware validation and save artifacts."""
    df = load_model_ready(input_path)
    if target != TARGET_COLUMN:
        logger.warning("Requested target '%s' differs from module target '%s'.", target, TARGET_COLUMN)

    X, y, feature_names = select_features(df)
    groups = df.loc[df[TARGET_COLUMN].notna(), "race_name"].reset_index(drop=True)
    train_idx, test_idx = race_aware_split(X, y, groups)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    race_test = groups.iloc[test_idx]
    logger.info(
        "Race-aware split: %d train rows (%d races), %d test rows (%d races).",
        len(train_idx), groups.iloc[train_idx].nunique(),
        len(test_idx), race_test.nunique(),
    )

    models = build_models()
    all_metrics = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        all_metrics[name] = evaluate(model, X_test, y_test)
        fitted[name] = model
        logger.info("%s: %s", name, {k: round(v, 4) for k, v in all_metrics[name].items()
                                     if isinstance(v, float)})

    best_name = max(all_metrics, key=lambda n: all_metrics[n]["f1"])
    best_model = fitted[best_name]
    logger.info("Best model by F1: %s", best_name)

    # Persist artifacts.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "pit_strategy_model.pkl")
    metadata = {
        "target": TARGET_COLUMN,
        "best_model": best_name,
        "features": feature_names,
        "validation": "GroupShuffleSplit(test_size=0.25) on race_name (race-aware)",
        "class_balance": y.value_counts(normalize=True).round(4).to_dict(),
        "n_rows": int(len(y)),
        "n_train_races": int(groups.iloc[train_idx].nunique()),
        "n_test_races": int(race_test.nunique()),
        "test_races": sorted(race_test.unique().tolist()),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    metrics_df = pd.DataFrame(all_metrics).T.reset_index().rename(columns={"index": "model"})
    metrics_df.to_csv(REPORTS_DIR / "global_model_metrics.csv", index=False)

    race_df = _race_level_metrics(best_model, X_test, y_test, race_test)
    race_df.insert(0, "model", best_name)
    race_df.to_csv(REPORTS_DIR / "race_level_model_metrics.csv", index=False)

    _plot_confusion(all_metrics[best_name], FIGURES_DIR / "confusion_matrix_global.png")
    _plot_model_comparison(all_metrics, FIGURES_DIR / "model_comparison.png")

    logger.info("Saved model + metadata to %s", MODEL_DIR)
    logger.info("Saved metrics and figures to %s / %s", REPORTS_DIR, FIGURES_DIR)
    return {"best_model": best_name, "metrics": all_metrics, "metadata": metadata}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train the global pit-strategy model.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--target", type=str, default=TARGET_COLUMN)
    args = parser.parse_args()

    input_path = PROCESSED_DIR / str(args.year) / "model_ready_all_tracks.csv"
    result = train_global_model(input_path, args.target, MODEL_DIR)
    print(f"Best model: {result['best_model']}")


if __name__ == "__main__":
    main()
