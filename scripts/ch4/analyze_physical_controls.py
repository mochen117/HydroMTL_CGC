# ==============================================================================
# Description:
#   Analyze physical controls on CGC transfer effects.
#
# Purpose:
#   Quantify monotonic relationships between basin attributes and streamflow NSE
#   gains of CGC relative to STL-Q. This script supports hydrological
#   interpretation of positive and negative transfer.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_with_metadata.csv
#
# Outputs:
#   - experiments/formal_ch4_training_experiments/summary/ch4_physical_control_correlation.csv
#   - experiments/formal_ch4_training_experiments/figures/fig4_9_physical_control_correlation_bar.png
#   - experiments/formal_ch4_training_experiments/figures/fig4_10_physical_control_scatter_matrix.png
# ==============================================================================

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CH3_SUMMARY_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling" / "06_summary"
CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
CH4_SUMMARY_DIR = CH4_DIR / "summary"
FIG_DIR = CH4_DIR / "figures"

INPUT_PATH = CH3_SUMMARY_DIR / "ch3_per_basin_with_metadata.csv"
OUTPUT_PATH = CH4_SUMMARY_DIR / "ch4_physical_control_correlation.csv"

DELTA_COL = "Delta_NSE_CGC_minus_STLQ"

ATTRIBUTE_ALIASES: Dict[str, List[str]] = {
    "aridity": ["aridity_index", "aridity"],
    "snow_fraction": ["snow_fraction", "frac_snow"],
    "area_gages2": ["area_gages2"],
    "elev_mean": ["elev_mean"],
    "slope_mean": ["slope_mean"],
    "frac_forest": ["frac_forest"],
    "lai_max": ["lai_max"],
    "soil_porosity": ["soil_porosity"],
    "soil_conductivity": ["soil_conductivity"],
    "sand_frac": ["sand_frac"],
    "clay_frac": ["clay_frac"],
}

ATTRIBUTE_LABELS: Dict[str, str] = {
    "aridity": "Aridity index",
    "snow_fraction": "Snow fraction",
    "area_gages2": "Drainage area",
    "elev_mean": "Mean elevation",
    "slope_mean": "Mean slope",
    "frac_forest": "Forest fraction",
    "lai_max": "Maximum LAI",
    "soil_porosity": "Soil porosity",
    "soil_conductivity": "Soil conductivity",
    "sand_frac": "Sand fraction",
    "clay_frac": "Clay fraction",
}

CH4_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def resolve_attribute_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    """Resolve an attribute column using accepted aliases."""
    for candidate in ATTRIBUTE_ALIASES.get(canonical_name, [canonical_name]):
        if candidate in df.columns:
            return candidate
    return None


def safe_corr(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """Compute Pearson and Spearman correlations safely."""
    valid = (
        pd.DataFrame({"x": x, "y": y})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(valid) < 5:
        return {
            "n": int(len(valid)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }

    pearson = pearsonr(valid["x"], valid["y"])
    spearman = spearmanr(valid["x"], valid["y"])

    return {
        "n": int(len(valid)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between basin attributes and CGC NSE gain."""
    records = []
    y = pd.to_numeric(df[DELTA_COL], errors="coerce")

    for canonical_attr in ATTRIBUTE_ALIASES:
        source_col = resolve_attribute_column(df, canonical_attr)
        if source_col is None:
            print(f"[Skip] Missing attribute: {canonical_attr}")
            continue

        x = pd.to_numeric(df[source_col], errors="coerce")
        corr = safe_corr(x, y)

        records.append(
            {
                "attribute": canonical_attr,
                "source_column": source_col,
                "label": ATTRIBUTE_LABELS.get(canonical_attr, canonical_attr),
                **corr,
                "abs_spearman_r": abs(corr["spearman_r"])
                if not pd.isna(corr["spearman_r"])
                else np.nan,
            }
        )

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values("abs_spearman_r", ascending=False).reset_index(drop=True)

    return out


def plot_correlation_bar(corr_df: pd.DataFrame) -> None:
    """Plot Spearman correlation ranking for physical controls."""
    if corr_df.empty:
        print("[Skip] Empty correlation table.")
        return

    plot_df = corr_df.dropna(subset=["spearman_r"]).sort_values("spearman_r")

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    bars = ax.barh(plot_df["label"], plot_df["spearman_r"])

    ax.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Spearman correlation with Delta NSE")
    ax.set_ylabel("Basin attribute")
    ax.set_xlim(-0.5, 0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.015 if width >= 0 else -0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=8,
        )

    output = FIG_DIR / "fig4_9_physical_control_correlation_bar.png"
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_selected_scatter(df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    """Plot scatter matrix for the top-ranked physical controls."""
    if corr_df.empty:
        return

    selected = corr_df.dropna(subset=["spearman_r"]).head(6)
    if selected.empty:
        print("[Skip] No selected attributes for scatter plots.")
        return

    ncols = 3
    nrows = int(np.ceil(len(selected) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)

    y = pd.to_numeric(df[DELTA_COL], errors="coerce").clip(-0.5, 0.5)

    for ax, (_, row) in zip(axes, selected.iterrows()):
        source_col = row["source_column"]
        label = row["label"]

        x = pd.to_numeric(df[source_col], errors="coerce")
        valid = pd.DataFrame({"x": x, "y": y}).dropna()

        ax.axhline(0.0, color="tab:blue", linestyle="--", linewidth=1.0)
        ax.scatter(valid["x"], valid["y"], s=16, alpha=0.65)
        ax.set_xlabel(label)
        ax.set_ylabel("Delta NSE")
        ax.set_ylim(-0.5, 0.5)
        ax.grid(True, linestyle="--", alpha=0.25)

    for ax in axes[len(selected):]:
        ax.axis("off")

    output = FIG_DIR / "fig4_10_physical_control_scatter_matrix.png"
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def main() -> None:
    """Run physical control analysis."""
    require_file(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, dtype={"gauge_id": str})

    if DELTA_COL not in df.columns:
        raise ValueError(f"Input table must contain '{DELTA_COL}'.")

    corr_df = compute_correlations(df)
    corr_df.to_csv(OUTPUT_PATH, index=False)

    plot_correlation_bar(corr_df)
    plot_selected_scatter(df, corr_df)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()