# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Audit Chapter 3 basin-level result tables for reproducibility and consistency.
#
# Purpose:
#   This script checks basin count, duplicate gauge IDs, required metric columns,
#   finite-value coverage, and Delta_NSE arithmetic consistency. It does not
#   modify model results.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_all_models.csv
#
# Outputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_result_audit_report.txt
#   - experiments/formal_ch3_modeling/06_summary/ch3_delta_consistency_issues.csv
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"

DEFAULT_INPUT = SUMMARY_DIR / "ch3_per_basin_all_models.csv"
DEFAULT_REPORT = SUMMARY_DIR / "ch3_result_audit_report.txt"
DEFAULT_DELTA_ISSUES = SUMMARY_DIR / "ch3_delta_consistency_issues.csv"

EXPECTED_BASIN_COUNT = 592
TOLERANCE = 1e-8

REQUIRED_COLUMNS = [
    "gauge_id",
    "STL_Q_streamflow_nse",
    "STL_ET_evapotranspiration_nse",
    "Hard_MTL_streamflow_nse",
    "Hard_MTL_evapotranspiration_nse",
    "MMoE_streamflow_nse",
    "MMoE_evapotranspiration_nse",
    "CGC_streamflow_nse",
    "CGC_evapotranspiration_nse",
    "Delta_NSE_HardMTL_minus_STLQ",
    "Delta_NSE_MMoE_minus_STLQ",
    "Delta_NSE_CGC_minus_STLQ",
    "Delta_NSE_HardMTL_ET_minus_STLET",
    "Delta_NSE_MMoE_ET_minus_STLET",
    "Delta_NSE_CGC_ET_minus_STLET",
]

DELTA_RULES: Dict[str, Tuple[str, str]] = {
    "Delta_NSE_HardMTL_minus_STLQ": (
        "Hard_MTL_streamflow_nse",
        "STL_Q_streamflow_nse",
    ),
    "Delta_NSE_MMoE_minus_STLQ": (
        "MMoE_streamflow_nse",
        "STL_Q_streamflow_nse",
    ),
    "Delta_NSE_CGC_minus_STLQ": (
        "CGC_streamflow_nse",
        "STL_Q_streamflow_nse",
    ),
    "Delta_NSE_HardMTL_ET_minus_STLET": (
        "Hard_MTL_evapotranspiration_nse",
        "STL_ET_evapotranspiration_nse",
    ),
    "Delta_NSE_MMoE_ET_minus_STLET": (
        "MMoE_evapotranspiration_nse",
        "STL_ET_evapotranspiration_nse",
    ),
    "Delta_NSE_CGC_ET_minus_STLET": (
        "CGC_evapotranspiration_nse",
        "STL_ET_evapotranspiration_nse",
    ),
}


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize gauge identifiers as 8-digit strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def metric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric metric columns excluding gauge_id."""
    return [col for col in df.columns if col != "gauge_id"]


def check_required_columns(df: pd.DataFrame) -> List[str]:
    """Return required columns missing from the result table."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def build_finite_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Build finite-value coverage summary for all metric columns."""
    records = []

    for col in metric_columns(df):
        values = pd.to_numeric(df[col], errors="coerce")
        finite = np.isfinite(values)
        records.append(
            {
                "column": col,
                "n_total": len(df),
                "n_finite": int(finite.sum()),
                "n_missing_or_nonfinite": int((~finite).sum()),
                "finite_rate_percent": float(finite.mean() * 100.0),
            }
        )

    return pd.DataFrame(records)


def check_delta_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Check whether stored Delta_NSE columns match model-minus-baseline values."""
    issues = []

    for delta_col, (model_col, base_col) in DELTA_RULES.items():
        if delta_col not in df.columns or model_col not in df.columns or base_col not in df.columns:
            continue

        delta = pd.to_numeric(df[delta_col], errors="coerce")
        model = pd.to_numeric(df[model_col], errors="coerce")
        base = pd.to_numeric(df[base_col], errors="coerce")
        expected = model - base

        valid = np.isfinite(delta) & np.isfinite(expected)
        diff = delta - expected
        bad = valid & (np.abs(diff) > TOLERANCE)

        if bad.any():
            tmp = df.loc[bad, ["gauge_id"]].copy()
            tmp["delta_column"] = delta_col
            tmp["model_column"] = model_col
            tmp["baseline_column"] = base_col
            tmp["stored_delta"] = delta[bad].values
            tmp["expected_delta"] = expected[bad].values
            tmp["absolute_error"] = np.abs(diff[bad]).values
            issues.append(tmp)

    if not issues:
        return pd.DataFrame(
            columns=[
                "gauge_id",
                "delta_column",
                "model_column",
                "baseline_column",
                "stored_delta",
                "expected_delta",
                "absolute_error",
            ]
        )

    return pd.concat(issues, ignore_index=True)


def write_report(
    path: Path,
    df: pd.DataFrame,
    missing_cols: List[str],
    duplicate_ids: pd.Series,
    coverage: pd.DataFrame,
    delta_issues: pd.DataFrame,
) -> None:
    """Write plain-text audit report."""
    lines = []
    lines.append("Chapter 3 result audit report")
    lines.append("=" * 80)
    lines.append(f"Input basin count          : {len(df)}")
    lines.append(f"Expected basin count       : {EXPECTED_BASIN_COUNT}")
    lines.append(f"Basin count status         : {'OK' if len(df) == EXPECTED_BASIN_COUNT else 'CHECK'}")
    lines.append(f"Unique gauge_id count      : {df['gauge_id'].nunique()}")
    lines.append(f"Duplicate gauge_id count   : {int(duplicate_ids.sum())}")
    lines.append("")

    lines.append("Required columns")
    lines.append("-" * 80)
    if missing_cols:
        lines.extend([f"MISSING: {col}" for col in missing_cols])
    else:
        lines.append("OK: all required columns are present.")
    lines.append("")

    lines.append("Finite-value coverage")
    lines.append("-" * 80)
    lines.append(coverage.to_string(index=False))
    lines.append("")

    lines.append("Delta_NSE consistency")
    lines.append("-" * 80)
    if delta_issues.empty:
        lines.append("OK: all available Delta_NSE columns match model-minus-baseline values.")
    else:
        lines.append(f"CHECK: {len(delta_issues)} inconsistent Delta_NSE records found.")
    lines.append("")

    if duplicate_ids.any():
        lines.append("Duplicate gauge IDs")
        lines.append("-" * 80)
        lines.append(df.loc[duplicate_ids, "gauge_id"].to_string(index=False))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run Chapter 3 result-table audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--delta_issues", type=Path, default=DEFAULT_DELTA_ISSUES)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, dtype={"gauge_id": str})
    if "gauge_id" not in df.columns:
        raise ValueError("Input table must contain a 'gauge_id' column.")

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    missing_cols = check_required_columns(df)
    duplicate_ids = df["gauge_id"].duplicated(keep=False)
    coverage = build_finite_coverage(df)
    delta_issues = check_delta_consistency(df)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.report, df, missing_cols, duplicate_ids, coverage, delta_issues)
    delta_issues.to_csv(args.delta_issues, index=False)

    print("Chapter 3 result audit completed.")
    print(f"Audit report       : {args.report}")
    print(f"Delta issue table  : {args.delta_issues}")
    print(f"Basin count        : {len(df)}")
    print(f"Missing columns    : {len(missing_cols)}")
    print(f"Delta issues       : {len(delta_issues)}")


if __name__ == "__main__":
    main()
