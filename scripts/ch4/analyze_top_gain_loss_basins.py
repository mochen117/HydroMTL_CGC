# ==============================================================================
# Description:
#   Analyze representative top-gain and top-loss basins for Chapter 4.
#
# Purpose:
#   Identify basins where CGC strongly improves or degrades streamflow
#   simulation relative to STL-Q, then export hydrological attributes for
#   interpretation and tabular reporting.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_with_metadata.csv
#
# Outputs:
#   - experiments/formal_ch4_training_experiments/summary/ch4_top_gain_basins_with_attributes.csv
#   - experiments/formal_ch4_training_experiments/summary/ch4_top_loss_basins_with_attributes.csv
#   - experiments/formal_ch4_training_experiments/summary/ch4_gain_loss_attribute_summary.csv
# ==============================================================================

from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CH3_SUMMARY_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling" / "06_summary"
CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
SUMMARY_DIR = CH4_DIR / "summary"

INPUT_PATH = CH3_SUMMARY_DIR / "ch3_per_basin_with_metadata.csv"
TOP_GAIN_PATH = SUMMARY_DIR / "ch4_top_gain_basins_with_attributes.csv"
TOP_LOSS_PATH = SUMMARY_DIR / "ch4_top_loss_basins_with_attributes.csv"
SUMMARY_PATH = SUMMARY_DIR / "ch4_gain_loss_attribute_summary.csv"

DELTA_COL = "Delta_NSE_CGC_minus_STLQ"
TOP_K = 50

ATTRIBUTE_COLUMNS: List[str] = [
    "gauge_id",
    "latitude",
    "longitude",
    "huc_02",
    "CGC_streamflow_nse",
    "STL-Q_streamflow_nse",
    DELTA_COL,
    "aridity",
    "aridity_index",
    "frac_snow",
    "snow_fraction",
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "area_gages2",
    "elev_mean",
    "slope_mean",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "soil_porosity",
    "soil_depth_statgso",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "clay_frac",
]

SUMMARY_ATTRIBUTES: List[str] = [
    "aridity",
    "aridity_index",
    "frac_snow",
    "snow_fraction",
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "area_gages2",
    "elev_mean",
    "slope_mean",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "soil_porosity",
    "soil_depth_statgso",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "clay_frac",
]

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize CAMELS gauge ids as eight-digit strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def summarize_attributes(
    df: pd.DataFrame,
    group_name: str,
    attributes: List[str],
) -> pd.DataFrame:
    """Summarize selected attributes for a basin group."""
    records = []

    for attr in attributes:
        if attr not in df.columns:
            continue

        values = pd.to_numeric(df[attr], errors="coerce").dropna()
        if values.empty:
            continue

        records.append(
            {
                "group": group_name,
                "attribute": attr,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Export representative top-gain and top-loss basins."""
    require_file(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, dtype={"gauge_id": str, "huc_02": str})

    if "gauge_id" not in df.columns:
        raise ValueError("Input table must contain 'gauge_id'.")
    if DELTA_COL not in df.columns:
        raise ValueError(f"Input table must contain '{DELTA_COL}'.")

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])
    df[DELTA_COL] = pd.to_numeric(df[DELTA_COL], errors="coerce")
    df = df.dropna(subset=[DELTA_COL]).copy()

    keep_cols = [col for col in ATTRIBUTE_COLUMNS if col in df.columns]

    top_gain = df.sort_values(DELTA_COL, ascending=False).head(TOP_K)[keep_cols]
    top_loss = df.sort_values(DELTA_COL, ascending=True).head(TOP_K)[keep_cols]

    top_gain.to_csv(TOP_GAIN_PATH, index=False)
    top_loss.to_csv(TOP_LOSS_PATH, index=False)

    summary = pd.concat(
        [
            summarize_attributes(top_gain, "Top gain", SUMMARY_ATTRIBUTES),
            summarize_attributes(top_loss, "Top loss", SUMMARY_ATTRIBUTES),
        ],
        ignore_index=True,
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Saved: {TOP_GAIN_PATH}")
    print(f"Saved: {TOP_LOSS_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()