# ==============================================================================
# Description:
#   Analyze CGC transfer effects across HUC2 hydrologic regions.
#
# Purpose:
#   Summarize streamflow NSE gains of CGC relative to STL-Q by HUC2 region.
#   This script supports Chapter 4 regional heterogeneity analysis under
#   different hydrological data conditions.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_with_metadata.csv
#
# Outputs:
#   - experiments/formal_ch4_basin_groups/summary/ch4_huc_group_summary.csv
#   - experiments/formal_ch4_basin_groups/figures/fig4_7_huc_group_delta_nse_boxplot.png
#   - experiments/formal_ch4_basin_groups/figures/fig4_8_huc_group_positive_rate.png
# ==============================================================================

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CH3_SUMMARY_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling" / "06_summary"
CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_basin_groups"
CH4_SUMMARY_DIR = CH4_DIR / "summary"
FIG_DIR = CH4_DIR / "figures"

INPUT_PATH = CH3_SUMMARY_DIR / "ch3_per_basin_with_metadata.csv"
SUMMARY_PATH = CH4_SUMMARY_DIR / "ch4_huc_group_summary.csv"

DELTA_COL = "Delta_NSE_CGC_minus_STLQ"
CGC_NSE_COL = "CGC_streamflow_nse"
STLQ_NSE_COL = "STL_Q_streamflow_nse"

MIN_BASINS_FOR_BOXPLOT = 3
DELTA_CLIP_RANGE = (-0.5, 0.5)

CH4_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_huc2(series: pd.Series) -> pd.Series:
    """Normalize HUC2 region codes as two-digit strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(2)
    )


def validate_columns(df: pd.DataFrame) -> None:
    """Validate all required input columns before analysis."""
    required_cols = [
        "huc_02",
        DELTA_COL,
        CGC_NSE_COL,
        STLQ_NSE_COL,
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in ch3_per_basin_with_metadata.csv: "
            f"{missing}"
        )


def summarize_huc_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize CGC transfer statistics by HUC2 region."""
    records = []

    for huc, sub in df.groupby("huc_02", dropna=True):
        delta = pd.to_numeric(sub[DELTA_COL], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if delta.empty:
            continue

        cgc_nse = pd.to_numeric(sub[CGC_NSE_COL], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        stlq_nse = pd.to_numeric(sub[STLQ_NSE_COL], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        records.append(
            {
                "huc_02": huc,
                "n_basins": int(len(delta)),
                "mean_delta_nse": float(delta.mean()),
                "median_delta_nse": float(delta.median()),
                "q25_delta_nse": float(delta.quantile(0.25)),
                "q75_delta_nse": float(delta.quantile(0.75)),
                "positive_transfer_count": int((delta > 0.0).sum()),
                "negative_transfer_count": int((delta < 0.0).sum()),
                "neutral_transfer_count": int((delta == 0.0).sum()),
                "positive_transfer_rate_pct": float((delta > 0.0).mean() * 100.0),
                "negative_transfer_rate_pct": float((delta < 0.0).mean() * 100.0),
                "strong_gain_rate_pct_delta_gt_0.05": float((delta > 0.05).mean() * 100.0),
                "strong_loss_rate_pct_delta_lt_minus_0.05": float((delta < -0.05).mean() * 100.0),
                "median_cgc_streamflow_nse": float(cgc_nse.median()) if not cgc_nse.empty else np.nan,
                "median_stlq_streamflow_nse": float(stlq_nse.median()) if not stlq_nse.empty else np.nan,
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("huc_02").reset_index(drop=True)


def plot_huc_delta_boxplot(df: pd.DataFrame) -> None:
    """Plot HUC2-wise distributions of CGC streamflow NSE gain."""
    hucs = sorted(df["huc_02"].dropna().unique())
    data: List[np.ndarray] = []
    labels: List[str] = []

    for huc in hucs:
        values = pd.to_numeric(
            df.loc[df["huc_02"] == huc, DELTA_COL],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) < MIN_BASINS_FOR_BOXPLOT:
            continue

        clipped = values.clip(*DELTA_CLIP_RANGE).to_numpy()
        data.append(clipped)
        labels.append(huc)

    if not data:
        print("[Skip] No valid HUC2 data for boxplot.")
        return

    fig, ax = plt.subplots(figsize=(11.0, 5.5))

    ax.axhline(0.0, color="tab:blue", linestyle="--", linewidth=1.0)
    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        widths=0.55,
        medianprops={"linewidth": 1.4},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )

    for idx, values in enumerate(data, start=1):
        median = float(np.median(values))
        offset = 0.018 if median <= 0.42 else -0.035
        va = "bottom" if offset > 0 else "top"

        ax.text(
            idx,
            median + offset,
            f"{median:.3f}",
            ha="center",
            va=va,
            fontsize=8,
            rotation=90,
        )

    ax.set_xlabel("HUC2 region")
    ax.set_ylabel("Delta NSE: CGC - STL-Q")
    ax.set_ylim(*DELTA_CLIP_RANGE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    output = FIG_DIR / "fig4_7_huc_group_delta_nse_boxplot.png"
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_huc_positive_rate(summary: pd.DataFrame) -> None:
    """Plot positive transfer rate by HUC2 region."""
    if summary.empty:
        print("[Skip] Empty HUC2 summary.")
        return

    plot_df = summary.sort_values("positive_transfer_rate_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    bars = ax.barh(plot_df["huc_02"], plot_df["positive_transfer_rate_pct"])

    ax.set_xlabel("Positive transfer rate (%)")
    ax.set_ylabel("HUC2 region")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar in bars:
        width = float(bar.get_width())
        ax.text(
            width + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=8,
        )

    output = FIG_DIR / "fig4_8_huc_group_positive_rate.png"
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def main() -> None:
    """Run HUC2 regional transfer analysis."""
    require_file(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, dtype={"gauge_id": str, "huc_02": str})
    validate_columns(df)

    df["huc_02"] = normalize_huc2(df["huc_02"])

    summary = summarize_huc_groups(df)
    summary.to_csv(SUMMARY_PATH, index=False)

    plot_huc_delta_boxplot(df)
    plot_huc_positive_rate(summary)

    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()