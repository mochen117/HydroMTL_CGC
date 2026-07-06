# ==============================================================================
# Description:
#   Analyze positive and negative transfer effects for Chapter 3.
#
# Purpose:
#   Build long-format and summary-format transfer tables for both streamflow
#   and evapotranspiration using standardized model names. Positive transfer
#   indicates that a multi-task model achieves higher NSE than the corresponding
#   single-task baseline for the same basin and task.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_all_models.csv
#
# Outputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_transfer_long.csv
#   - experiments/formal_ch3_modeling/06_summary/ch3_transfer_summary.csv
# ==============================================================================

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling" / "06_summary"

INPUT_PATH = SUMMARY_DIR / "ch3_per_basin_all_models.csv"
TRANSFER_LONG_PATH = SUMMARY_DIR / "ch3_transfer_long.csv"
TRANSFER_SUMMARY_PATH = SUMMARY_DIR / "ch3_transfer_summary.csv"


TRANSFER_CONFIG: Dict[str, Dict[str, object]] = {
    "streamflow": {
        "baseline_model": "STL_Q",
        "baseline_col": "STL_Q_streamflow_nse",
        "models": {
            "Hard_MTL": "Hard_MTL_streamflow_nse",
            "MMoE": "MMoE_streamflow_nse",
            "CGC": "CGC_streamflow_nse",
        },
    },
    "evapotranspiration": {
        "baseline_model": "STL_ET",
        "baseline_col": "STL_ET_evapotranspiration_nse",
        "models": {
            "Hard_MTL": "Hard_MTL_evapotranspiration_nse",
            "MMoE": "MMoE_evapotranspiration_nse",
            "CGC": "CGC_evapotranspiration_nse",
        },
    },
}


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize CAMELS gauge ids as 8-digit strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def safe_rate(mask: pd.Series) -> float:
    """Return percentage rate for a Boolean mask."""
    if len(mask) == 0:
        return np.nan
    return float(mask.mean() * 100.0)


def classify_transfer(delta: float) -> str:
    """Classify basin-level transfer effect."""
    if pd.isna(delta):
        return "missing"
    if delta > 0.0:
        return "positive"
    if delta < 0.0:
        return "negative"
    return "neutral"


def validate_columns(df: pd.DataFrame) -> None:
    """Validate required columns before analysis."""
    required_cols = ["gauge_id"]

    for task_config in TRANSFER_CONFIG.values():
        required_cols.append(str(task_config["baseline_col"]))
        required_cols.extend(task_config["models"].values())

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns in ch3_per_basin_all_models.csv: "
            f"{missing}"
        )


def build_transfer_long(df: pd.DataFrame) -> pd.DataFrame:
    """Build long-format basin-level transfer table for all tasks."""
    records: List[Dict[str, object]] = []

    for task_name, task_config in TRANSFER_CONFIG.items():
        baseline_model = str(task_config["baseline_model"])
        baseline_col = str(task_config["baseline_col"])
        model_map: Dict[str, str] = task_config["models"]  # type: ignore[assignment]

        for model_name, model_col in model_map.items():
            valid = df[["gauge_id", baseline_col, model_col]].copy()

            valid[baseline_col] = pd.to_numeric(valid[baseline_col], errors="coerce")
            valid[model_col] = pd.to_numeric(valid[model_col], errors="coerce")
            valid = valid.replace([np.inf, -np.inf], np.nan).dropna()

            valid["delta_nse"] = valid[model_col] - valid[baseline_col]

            for _, row in valid.iterrows():
                delta = float(row["delta_nse"])

                records.append(
                    {
                        "gauge_id": row["gauge_id"],
                        "task": task_name,
                        "model": model_name,
                        "baseline_model": baseline_model,
                        "baseline_nse": float(row[baseline_col]),
                        "model_nse": float(row[model_col]),
                        "delta_nse": delta,
                        "transfer_type": classify_transfer(delta),
                    }
                )

    return pd.DataFrame(records)


def summarize_transfer(transfer_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize transfer effects by task and model."""
    records: List[Dict[str, object]] = []

    grouped = transfer_df.groupby(["task", "model", "baseline_model"], sort=True)

    for (task_name, model_name, baseline_model), group in grouped:
        delta = pd.to_numeric(group["delta_nse"], errors="coerce").dropna()

        if delta.empty:
            continue

        records.append(
            {
                "task": task_name,
                "model": model_name,
                "baseline_model": baseline_model,
                "n_basins": int(len(delta)),
                "mean_delta_nse": float(delta.mean()),
                "median_delta_nse": float(delta.median()),
                "q25_delta_nse": float(delta.quantile(0.25)),
                "q75_delta_nse": float(delta.quantile(0.75)),
                "positive_transfer_count": int((delta > 0.0).sum()),
                "negative_transfer_count": int((delta < 0.0).sum()),
                "neutral_transfer_count": int((delta == 0.0).sum()),
                "positive_transfer_rate_pct": safe_rate(delta > 0.0),
                "negative_transfer_rate_pct": safe_rate(delta < 0.0),
                "neutral_transfer_rate_pct": safe_rate(delta == 0.0),
                "strong_gain_count_delta_gt_0.05": int((delta > 0.05).sum()),
                "strong_loss_count_delta_lt_minus_0.05": int((delta < -0.05).sum()),
                "strong_gain_rate_pct_delta_gt_0.05": safe_rate(delta > 0.05),
                "strong_loss_rate_pct_delta_lt_minus_0.05": safe_rate(delta < -0.05),
            }
        )

    return pd.DataFrame(records)


def print_validation_summary(transfer_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Print a compact validation summary."""
    print("=" * 100)
    print("Chapter 3 transfer analysis validation")

    if transfer_df.empty:
        raise ValueError("Transfer long table is empty.")

    task_counts = transfer_df.groupby("task").size().to_dict()
    model_counts = transfer_df.groupby(["task", "model"]).size().to_dict()

    print(f"Long-format records: {len(transfer_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print(f"Records by task: {task_counts}")
    print(f"Records by task and model: {model_counts}")
    print("Validation passed.")
    print("=" * 100)


def main() -> None:
    """Run Chapter 3 transfer-effect analysis."""
    require_file(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, dtype={"gauge_id": str})

    if "gauge_id" not in df.columns:
        raise ValueError("Input table must contain 'gauge_id'.")

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    validate_columns(df)

    transfer_df = build_transfer_long(df)
    summary_df = summarize_transfer(transfer_df)

    transfer_df.to_csv(TRANSFER_LONG_PATH, index=False)
    summary_df.to_csv(TRANSFER_SUMMARY_PATH, index=False)

    print_validation_summary(transfer_df, summary_df)

    print(f"Saved: {TRANSFER_LONG_PATH}")
    print(f"Saved: {TRANSFER_SUMMARY_PATH}")


if __name__ == "__main__":
    main()