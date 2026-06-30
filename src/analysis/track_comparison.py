"""Track comparison analysis for the 2025 season.

Computes how pit strategy, stint behaviour, tyre usage and safety-car/VSC
periods differ across races, saves charts to
``outputs/figures/track_comparison/`` and a Markdown summary to
``outputs/reports/track_comparison_summary.md``.

The track groupings below are **analysis labels only** — they are not used by
the core data pipeline or the model.

CLI::

    python src/analysis/track_comparison.py --year 2025
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys as _sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))

from src import plotting as pal  # noqa: E402
from src.plotting import apply_dark_theme  # noqa: E402
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "track_comparison"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("track_comparison")

# Analysis-only labels (NOT used by the pipeline or model).
TRACK_GROUPS: dict[str, list[str]] = {
    "street": ["Monaco", "Singapore", "Azerbaijan", "Las Vegas", "Saudi Arabia"],
    "high_degradation": ["Bahrain", "Spain", "Hungary", "Qatar"],
    "fast_flowing": ["Great Britain", "Belgium", "Italy", "Japan"],
    "mixed": ["Canada", "Austria", "Mexico City", "São Paulo", "Abu Dhabi"],
}


def _group_label(race: str) -> str:
    for label, races in TRACK_GROUPS.items():
        if race in races:
            return label
    return "other"


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #
def pit_stops_by_race(df: pd.DataFrame) -> pd.DataFrame:
    """Average number of pit stops per driver, by race."""
    pit = df[df["pit_in_time"].notna()] if "pit_in_time" in df.columns else df[df["is_pit_lap"]]
    per_driver = pit.groupby(["race_name", "driver"]).size().reset_index(name="stops")
    out = per_driver.groupby("race_name")["stops"].mean().reset_index(name="avg_pit_stops")
    return out.sort_values("avg_pit_stops", ascending=False)


def stint_length_by_race(df: pd.DataFrame) -> pd.DataFrame:
    """Average stint length (laps) by race."""
    stint_lengths = (
        df.groupby(["race_name", "driver", "stint"]).size().reset_index(name="laps")
    )
    out = stint_lengths.groupby("race_name")["laps"].mean().reset_index(name="avg_stint_length")
    return out.sort_values("avg_stint_length", ascending=False)


def stint_length_by_compound_race(df: pd.DataFrame) -> pd.DataFrame:
    """Average stint length by compound and race."""
    stint_lengths = (
        df.groupby(["race_name", "driver", "stint", "compound"]).size().reset_index(name="laps")
    )
    return (
        stint_lengths.groupby(["race_name", "compound"])["laps"].mean().reset_index(
            name="avg_stint_length"
        )
    )


def compound_usage_by_race(df: pd.DataFrame) -> pd.DataFrame:
    """Share of laps run on each compound, by race."""
    counts = df.groupby(["race_name", "compound"]).size().reset_index(name="laps")
    totals = counts.groupby("race_name")["laps"].transform("sum")
    counts["share"] = counts["laps"] / totals
    return counts


def pit_lap_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Pit-in lap numbers (as race percentage) for distribution plots."""
    pit = df[df["pit_in_time"].notna()] if "pit_in_time" in df.columns else df[df["is_pit_lap"]]
    cols = ["race_name", "driver", "lap_number"]
    out = pit[cols].copy()
    if "lap_percentage" not in out.columns:
        total = df.groupby("race_name")["lap_number"].max()
        out["lap_percentage"] = out.apply(
            lambda r: r["lap_number"] / total[r["race_name"]], axis=1
        )
    else:
        out = pit[cols + ["lap_percentage"]].copy()
    return out


def sc_vsc_pit_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Share of pit-ins that occur during SC/VSC vs green, by race."""
    pit = df[df["pit_in_time"].notna()].copy() if "pit_in_time" in df.columns else df[df["is_pit_lap"]].copy()
    if pit.empty:
        return pd.DataFrame(columns=["race_name", "pit_under_sc_vsc_share"])
    pit["under_sc_vsc"] = pit["safety_car_or_vsc_flag"].astype(bool)
    out = pit.groupby("race_name")["under_sc_vsc"].mean().reset_index(
        name="pit_under_sc_vsc_share"
    )
    return out.sort_values("pit_under_sc_vsc_share", ascending=False)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _barh(data: pd.DataFrame, value_col: str, title: str, xlabel: str, path: Path) -> None:
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.36 * len(data))))
    ax.barh(data["race_name"], data[value_col], color=pal.ACCENT)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_track_comparison(
    laps_path: str | Path,
    fig_dir: str | Path = FIGURES_DIR,
    report_path: str | Path = REPORTS_DIR / "track_comparison_summary.md",
    race_metrics_path: str | Path | None = None,
) -> None:
    """Generate all track-comparison figures and the Markdown summary."""
    laps_path = Path(laps_path)
    if not laps_path.is_file():
        raise FileNotFoundError(f"Laps dataset not found: {laps_path}")
    df = pd.read_csv(laps_path)
    if df.empty:
        logger.warning("Empty laps dataset; nothing to analyze.")
        return

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stops = pit_stops_by_race(df)
    stints = stint_length_by_race(df)
    compound = compound_usage_by_race(df)
    pit_dist = pit_lap_distribution(df)
    sc_impact = sc_vsc_pit_impact(df)

    _barh(stops, "avg_pit_stops", "Average pit stops per driver by race",
          "Avg pit stops", fig_dir / "pit_stop_count_by_race.png")
    _barh(stints, "avg_stint_length", "Average stint length by race",
          "Avg stint length (laps)", fig_dir / "avg_stint_length_by_race.png")

    # Compound usage: stacked bar.
    pivot = compound.pivot_table(index="race_name", columns="compound", values="share",
                                 fill_value=0)
    # Canonical F1 compound colours (also on-brand: SOFT is red).
    compound_colors = {
        "SOFT": "#E10600", "MEDIUM": "#F7C82E", "HARD": "#EDEDED",
        "INTERMEDIATE": "#3BB143", "WET": "#2E6FE0", "UNKNOWN": "#7A7F8A",
    }
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[compound_colors.get(c, "#7A7F8A") for c in pivot.columns])
    ax.set_xlabel("Share of laps"); ax.set_title("Tyre compound usage by race")
    ax.grid(axis="y", visible=False)
    leg = ax.legend(title="compound", bbox_to_anchor=(1.02, 1), loc="upper left")
    leg.get_title().set_color(pal.INK)
    fig.tight_layout(); fig.savefig(fig_dir / "compound_usage_by_race.png")
    plt.close(fig)

    # Pit lap distribution histogram (race percentage).
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(pit_dist["lap_percentage"].dropna(), bins=20, color=pal.ACCENT,
            edgecolor=pal.BG)
    ax.set_xlabel("Race completion at pit-in (fraction)")
    ax.set_ylabel("Number of pit stops")
    ax.set_title("Pit-in timing distribution (all races)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(fig_dir / "pit_lap_distribution_by_race.png")
    plt.close(fig)

    # Race-level model performance (if metrics available).
    race_metrics = None
    if race_metrics_path is None:
        race_metrics_path = REPORTS_DIR / "race_level_model_metrics.csv"
    race_metrics_path = Path(race_metrics_path)
    if race_metrics_path.is_file():
        race_metrics = pd.read_csv(race_metrics_path)
        if not race_metrics.empty:
            rm = race_metrics.sort_values("f1", ascending=False)
            apply_dark_theme()
            fig, ax = plt.subplots(figsize=(7.5, max(3, 0.42 * len(rm))))
            ax.barh(rm["race_name"], rm["f1"], color=pal.ACCENT)
            ax.invert_yaxis(); ax.set_xlabel("F1 (test races)")
            ax.set_title("Race-level model performance (held-out)")
            ax.grid(axis="y", visible=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "race_level_model_performance.png")
            plt.close(fig)

    _write_report(report_path, df, stops, stints, compound, sc_impact, race_metrics)
    logger.info("Track comparison complete: figures -> %s", fig_dir)


def _write_report(report_path, df, stops, stints, compound, sc_impact, race_metrics) -> None:
    lines = ["# Track Comparison Summary (2025)", ""]
    lines.append(f"- Races analyzed: **{df['race_name'].nunique()}**")
    lines.append(f"- Total laps: **{len(df):,}**")
    lines.append(f"- Drivers: **{df['driver'].nunique()}**")
    lines.append("")
    lines.append("> Track groupings are analysis labels only and are not used by the data "
                 "pipeline or model.")
    lines.append("")

    lines.append("## Average pit stops per driver (top 5)")
    lines.append(stops.head(5).to_markdown(index=False))
    lines.append("")
    lines.append("## Average stint length by race (top 5 longest)")
    lines.append(stints.head(5).to_markdown(index=False))
    lines.append("")

    lines.append("## Pit stops under safety car / VSC (top 5 share)")
    lines.append(sc_impact.head(5).to_markdown(index=False))
    lines.append("")

    lines.append("## Track groupings (analysis labels)")
    for label, races in TRACK_GROUPS.items():
        present = [r for r in races if r in set(df["race_name"])]
        lines.append(f"- **{label}**: {', '.join(present)}")
    lines.append("")

    if race_metrics is not None and not race_metrics.empty:
        lines.append("## Race-level model performance (held-out test races)")
        lines.append("Model performance is reported only for races held out of training "
                     "(race-aware split), so coverage is a subset of all races.")
        lines.append("")
        lines.append(race_metrics.sort_values("f1", ascending=False).to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## Race-level model performance")
        lines.append("_No race-level model metrics found. Run the training script first._")
        lines.append("")

    lines.append("## Figures")
    for fig in ["pit_stop_count_by_race.png", "avg_stint_length_by_race.png",
                "compound_usage_by_race.png", "pit_lap_distribution_by_race.png",
                "race_level_model_performance.png"]:
        lines.append(f"- `outputs/figures/track_comparison/{fig}`")
    lines.append("")

    Path(report_path).write_text("\n".join(lines))
    logger.info("Wrote report -> %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run track comparison analysis.")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    laps_path = PROCESSED_DIR / str(args.year) / "laps_all_tracks.csv"
    run_track_comparison(laps_path)
    print(REPORTS_DIR / "track_comparison_summary.md")


if __name__ == "__main__":
    main()
