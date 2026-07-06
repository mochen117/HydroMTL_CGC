# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Diagnose extreme NSE values in Chapter 3 basin-level model results.
#
# Purpose:
#   This script does not alter model outputs. It summarizes NSE distributions,
#   counts extreme negative values, and exports basin lists requiring inspection.
#   The outputs support transparent reporting and figure annotation.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_all_models.csv
#
# Outputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_nse_diagnostics_summary.csv
#   - experiments/formal_ch3_modeling/06_summary/ch3_nse_outlier_basins.csv
#   - experiments/formal_ch3_modeling/06_summary/ch3_nse_outlier_overlap.csv
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"

DEFAULT_INPUT = SUMMARY_DIR / "ch3_per_basin_all_models.csv"
DEFAULT_SUMMARY_OUTPUT = SUMMARY_DIR / "ch3_nse_diagnostics_summary.csv"
DEFAULT_OUTLIER_OUTPUT = SUMMARY_DIR / "ch3_nse_outlier_basins.csv"
DEFAULT_OVERLAP_OUTPUT = SUMMARY_DIR / "ch3_nse_outlier_overlap.csv"

NSE_THRESHOLDS = [-1.0, -10.0, -100.0]


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize gauge identifiers as 8-digit strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def nse_columns(df: pd.DataFrame) -> List[str]:
    """Return all NSE metric columns except derived delta columns."""
    return [
        col for col in df.columns
        if col.lower().endswith("_nse") and not col.lower().startswith("delta_")
    ]


def infer_task(column: str) -> str:
    """Infer hydrological task from a metric column name."""
    lower = column.lower()
    if "streamflow" in lower:
        return "streamflow"
    if "evapotranspiration" in lower:
        return "evapotranspiration"
    return "unknown"


def finite_values(series: pd.Series) -> pd.Series:
    """Return finite numeric values only."""
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def build_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Build column-level NSE diagnostics."""
    records: List[Dict[str, object]] = []

    for col in cols:
        values = finite_values(df[col])
        total = len(df)

        if values.empty:
            records.append(
                {
                    "metric_column": col,
                    "task": infer_task(col),
                    "n_total_basins": total,
                    "n_finite": 0,
                    "n_missing": total,
                    "min": np.nan,
                    "q01": np.nan,
                    "q05": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "q95": np.nan,
                    "q99": np.nan,
                    "max": np.nan,
                    "n_nse_lt_-1": 0,
                    "n_nse_lt_-10": 0,
                    "n_nse_lt_-100": 0,
                    "rate_nse_lt_-1_percent": 0.0,
                    "rate_nse_lt_-10_percent": 0.0,
                    "rate_nse_lt_-100_percent": 0.0,
                }
            )
            continue

        rec: Dict[str, object] = {
            "metric_column": col,
            "task": infer_task(col),
            "n_total_basins": total,
            "n_finite": int(values.size),
            "n_missing": int(total - values.size),
            "min": float(values.min()),
            "q01": float(values.quantile(0.01)),
            "q05": float(values.quantile(0.05)),
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
            "q95": float(values.quantile(0.95)),
            "q99": float(values.quantile(0.99)),
            "max": float(values.max()),
        }

        for threshold in NSE_THRESHOLDS:
            count = int((values < threshold).sum())
            key = str(int(abs(threshold)))
            rec[f"n_nse_lt_-{key}"] = count
            rec[f"rate_nse_lt_-{key}_percent"] = float(count / values.size * 100.0)

        records.append(rec)

    return pd.DataFrame(records)


def build_outlier_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Build basin-level table for all NSE values below the configured thresholds."""
    records: List[Dict[str, object]] = []

    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")

        for threshold in NSE_THRESHOLDS:
            mask = np.isfinite(values) & (values < threshold)
            subset = df.loc[mask, ["gauge_id"]].copy()
            subset["metric_column"] = col
            subset["task"] = infer_task(col)
            subset["threshold"] = threshold
            subset["nse_value"] = values[mask].values
            records.extend(subset.to_dict("records"))

    if not records:
        return pd.DataFrame(
            columns=["gauge_id", "metric_column", "task", "threshold", "nse_value"]
        )

    outliers = pd.DataFrame(records)
    outliers = outliers.sort_values(
        by=["threshold", "task", "metric_column", "nse_value"],
        ascending=[True, True, True, True],
    )
    return outliers


def build_overlap_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Summarize how many NSE columns are extreme for each basin."""
    result = df[["gauge_id"]].copy()

    for threshold in NSE_THRESHOLDS:
        flag_cols = []
        suffix = str(int(abs(threshold)))

        for col in cols:
            flag_col = f"{col}_lt_-{suffix}"
            result[flag_col] = pd.to_numeric(df[col], errors="coerce") < threshold
            flag_cols.append(flag_col)

        result[f"n_metrics_lt_-{suffix}"] = result[flag_cols].sum(axis=1)

    count_cols = [c for c in result.columns if c.startswith("n_metrics_lt_")]
    result = result.loc[result[count_cols].sum(axis=1) > 0].copy()
    result = result.sort_values(by=count_cols, ascending=False)

    return result


def main() -> None:
    """Run NSE outlier diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary_output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--outlier_output", type=Path, default=DEFAULT_OUTLIER_OUTPUT)
    parser.add_argument("--overlap_output", type=Path, default=DEFAULT_OVERLAP_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, dtype={"gauge_id": str})

    if "gauge_id" not in df.columns:
        raise ValueError("Input table must contain a 'gauge_id' column.")

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    cols = nse_columns(df)
    if not cols:
        raise ValueError("No NSE columns were found in the input table.")

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(df, cols)
    outliers = build_outlier_table(df, cols)
    overlap = build_overlap_table(df, cols)

    summary.to_csv(args.summary_output, index=False)
    outliers.to_csv(args.outlier_output, index=False)
    overlap.to_csv(args.overlap_output, index=False)

    print("NSE diagnostics completed.")
    print(f"Input basins        : {len(df)}")
    print(f"NSE columns         : {len(cols)}")
    print(f"Summary output      : {args.summary_output}")
    print(f"Outlier output      : {args.outlier_output}")
    print(f"Overlap output      : {args.overlap_output}")

    display_cols = [
        "metric_column",
        "n_finite",
        "min",
        "q01",
        "median",
        "n_nse_lt_-1",
        "n_nse_lt_-10",
        "n_nse_lt_-100",
    ]
    print("\nKey diagnostics:")
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()