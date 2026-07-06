# ==============================================================================
# Description:
#   Run Chapter 3 sensitivity analysis for extreme NSE values.
#
# Purpose:
#   Compare the main all-basin conclusions with a diagnostic subset after
#   excluding severely failed basin-task cases. This script does not modify any
#   experiment outputs and should be used only as a post-hoc robustness check.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_all_models.csv
#
# Outputs:
#   - ch3_sensitivity_metric_summary.csv
#   - ch3_sensitivity_transfer_summary.csv
#   - ch3_sensitivity_basins_removed.csv
#   - ch3_sensitivity_report.txt
# ==============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"

PER_BASIN_PATH = SUMMARY_DIR / "ch3_per_basin_all_models.csv"

METRIC_OUT = SUMMARY_DIR / "ch3_sensitivity_metric_summary.csv"
TRANSFER_OUT = SUMMARY_DIR / "ch3_sensitivity_transfer_summary.csv"
REMOVED_OUT = SUMMARY_DIR / "ch3_sensitivity_basins_removed.csv"
REPORT_OUT = SUMMARY_DIR / "ch3_sensitivity_report.txt"

NSE_THRESHOLD = -1.0

MODELS_Q = ["STL_Q", "Hard_MTL", "MMoE", "CGC"]
MODELS_ET = ["STL_ET", "Hard_MTL", "MMoE", "CGC"]
MTL_MODELS = ["Hard_MTL", "MMoE", "CGC"]

DISPLAY_LABELS = {
    "STL_Q": "STL-Q",
    "STL_ET": "STL-ET",
    "Hard_MTL": "Hard-MTL",
    "MMoE": "MMoE",
    "CGC": "CGC",
}

TASK_CONFIG = {
    "streamflow": {
        "models": MODELS_Q,
        "baseline": "STL_Q",
        "delta_cols": {
            "Hard_MTL": "Delta_NSE_HardMTL_minus_STLQ",
            "MMoE": "Delta_NSE_MMoE_minus_STLQ",
            "CGC": "Delta_NSE_CGC_minus_STLQ",
        },
    },
    "evapotranspiration": {
        "models": MODELS_ET,
        "baseline": "STL_ET",
        "delta_cols": {
            "Hard_MTL": "Delta_NSE_HardMTL_ET_minus_STLET",
            "MMoE": "Delta_NSE_MMoE_ET_minus_STLET",
            "CGC": "Delta_NSE_CGC_ET_minus_STLET",
        },
    },
}


def require_file(path: Path) -> None:
    """Raise an explicit error if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize gauge identifiers to 8-character CAMELS strings."""
    return series.astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(8)


def metric_column(model: str, task: str, metric: str) -> str:
    """Build the standardized metric column name."""
    return f"{model}_{task}_{metric}"


def clean_numeric(series: pd.Series) -> pd.Series:
    """Return finite numeric values only."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.dropna()


def load_table() -> pd.DataFrame:
    """Load the basin-level result table."""
    require_file(PER_BASIN_PATH)
    df = pd.read_csv(PER_BASIN_PATH, dtype={"gauge_id": str})
    if "gauge_id" not in df.columns:
        raise ValueError("Input table must contain 'gauge_id'.")
    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])
    return df


def task_valid_mask(df: pd.DataFrame, task: str, threshold: float) -> pd.Series:
    """
    Return basins whose NSE values are above threshold for all models of one task.

    The mask is intentionally task-specific. A basin removed for ET sensitivity
    is not automatically removed from streamflow sensitivity, and vice versa.
    """
    models = TASK_CONFIG[task]["models"]
    nse_cols = [metric_column(model, task, "nse") for model in models if metric_column(model, task, "nse") in df.columns]

    if not nse_cols:
        raise ValueError(f"No NSE columns found for task={task}.")

    values = df[nse_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = values.notna().all(axis=1)
    threshold_mask = (values > threshold).all(axis=1)
    return finite_mask & threshold_mask


def summarize_metrics(
    df: pd.DataFrame,
    task: str,
    subset_name: str,
    mask: pd.Series,
) -> List[Dict[str, object]]:
    """Summarize median and IQR of NSE and KGE for a given subset."""
    records: List[Dict[str, object]] = []
    models = TASK_CONFIG[task]["models"]

    for model in models:
        for metric in ["nse", "kge"]:
            col = metric_column(model, task, metric)
            if col not in df.columns:
                continue

            values = clean_numeric(df.loc[mask, col])
            if values.empty:
                records.append(
                    {
                        "subset": subset_name,
                        "task": task,
                        "model": DISPLAY_LABELS.get(model, model),
                        "metric": metric.upper(),
                        "n_basins": 0,
                        "median": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                        "iqr": np.nan,
                    }
                )
                continue

            q25 = float(values.quantile(0.25))
            q75 = float(values.quantile(0.75))
            records.append(
                {
                    "subset": subset_name,
                    "task": task,
                    "model": DISPLAY_LABELS.get(model, model),
                    "metric": metric.upper(),
                    "n_basins": int(values.shape[0]),
                    "median": float(values.median()),
                    "q25": q25,
                    "q75": q75,
                    "iqr": q75 - q25,
                }
            )

    return records


def summarize_transfer(
    df: pd.DataFrame,
    task: str,
    subset_name: str,
    mask: pd.Series,
) -> List[Dict[str, object]]:
    """Summarize paired NSE transfer under a given subset."""
    records: List[Dict[str, object]] = []
    delta_cols = TASK_CONFIG[task]["delta_cols"]

    for model, col in delta_cols.items():
        if col not in df.columns:
            continue

        values = clean_numeric(df.loc[mask, col])
        if values.empty:
            records.append(
                {
                    "subset": subset_name,
                    "task": task,
                    "model": DISPLAY_LABELS.get(model, model),
                    "n_basins": 0,
                    "median_delta_nse": np.nan,
                    "positive_rate": np.nan,
                    "negative_rate": np.nan,
                }
            )
            continue

        records.append(
            {
                "subset": subset_name,
                "task": task,
                "model": DISPLAY_LABELS.get(model, model),
                "n_basins": int(values.shape[0]),
                "median_delta_nse": float(values.median()),
                "positive_rate": float((values > 0.0).mean() * 100.0),
                "negative_rate": float((values < 0.0).mean() * 100.0),
            }
        )

    return records


def collect_removed_basins(df: pd.DataFrame, task: str, valid_mask: pd.Series) -> pd.DataFrame:
    """Collect basins removed by the task-specific sensitivity filter."""
    models = TASK_CONFIG[task]["models"]
    nse_cols = [metric_column(model, task, "nse") for model in models if metric_column(model, task, "nse") in df.columns]

    removed = df.loc[~valid_mask, ["gauge_id"] + nse_cols].copy()
    removed.insert(1, "task", task)
    removed.insert(2, "reason", f"At least one {task} NSE <= {NSE_THRESHOLD} or non-finite")
    return removed


def write_report(
    df: pd.DataFrame,
    metric_summary: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    removed: pd.DataFrame,
) -> None:
    """Write a compact text report for thesis documentation."""
    lines: List[str] = []
    lines.append("Chapter 3 sensitivity analysis")
    lines.append("=" * 80)
    lines.append(f"Input table: {PER_BASIN_PATH}")
    lines.append(f"Total basins in main table: {df['gauge_id'].nunique()}")
    lines.append(f"Task-specific sensitivity threshold: all model NSE values must be > {NSE_THRESHOLD}.")
    lines.append("")
    lines.append("Important interpretation")
    lines.append("- Main results should remain based on all 592 basins.")
    lines.append("- This sensitivity analysis is a robustness check, not a replacement for the main result.")
    lines.append("- Removed basins are task-specific; streamflow and evapotranspiration filters are not forced to be identical.")
    lines.append("")

    for task in TASK_CONFIG:
        task_removed = removed[removed["task"] == task]
        lines.append(f"{task}: removed basins = {len(task_removed)}")
    lines.append("")

    lines.append("Metric summary")
    lines.append(metric_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append("Transfer summary")
    lines.append(transfer_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run sensitivity analysis using existing Chapter 3 results."""
    df = load_table()

    all_metric_records: List[Dict[str, object]] = []
    all_transfer_records: List[Dict[str, object]] = []
    removed_frames: List[pd.DataFrame] = []

    all_mask = pd.Series(True, index=df.index)

    for task in TASK_CONFIG:
        sens_mask = task_valid_mask(df, task, NSE_THRESHOLD)

        all_metric_records.extend(summarize_metrics(df, task, "all_basins", all_mask))
        all_metric_records.extend(summarize_metrics(df, task, "nse_gt_minus_1", sens_mask))

        all_transfer_records.extend(summarize_transfer(df, task, "all_basins", all_mask))
        all_transfer_records.extend(summarize_transfer(df, task, "nse_gt_minus_1", sens_mask))

        removed_frames.append(collect_removed_basins(df, task, sens_mask))

    metric_summary = pd.DataFrame(all_metric_records)
    transfer_summary = pd.DataFrame(all_transfer_records)
    removed = pd.concat(removed_frames, ignore_index=True)

    metric_summary.to_csv(METRIC_OUT, index=False)
    transfer_summary.to_csv(TRANSFER_OUT, index=False)
    removed.to_csv(REMOVED_OUT, index=False)
    write_report(df, metric_summary, transfer_summary, removed)

    print(f"Saved: {METRIC_OUT}")
    print(f"Saved: {TRANSFER_OUT}")
    print(f"Saved: {REMOVED_OUT}")
    print(f"Saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()
