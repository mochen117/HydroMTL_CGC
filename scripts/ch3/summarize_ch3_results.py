# ==============================================================================
# Description:
#   Summarize formal Chapter 3 model results.
#
# Purpose:
#   Collect model-level validation summaries and basin-level validation metrics
#   for Chapter 3. This script standardizes model names, exports clean
#   basin-level metric tables, and computes task-specific NSE gain columns for
#   both streamflow and evapotranspiration.
#
# Outputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_performance_summary.csv
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_all_models.csv
# ==============================================================================

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


MODEL_DIRS: Dict[str, Path] = {
    "STL_Q": CH3_DIR / "01_stl_q" / "ch3_stl_q_seed42",
    "STL_ET": CH3_DIR / "02_stl_et" / "ch3_stl_et_seed42",
    "Hard_MTL": CH3_DIR / "03_hard_mtl" / "ch3_hard_mtl_seed42",
    "MMoE": CH3_DIR / "04_mmoe_mtl" / "ch3_mmoe_mtl_seed42",
    "CGC": CH3_DIR / "05_cgc_mtl" / "ch3_cgc_mtl_seed42",
}


GAIN_DEFINITIONS: Dict[str, Tuple[str, str]] = {
    # Streamflow gains relative to STL-Q.
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
    "Delta_NSE_MMoE_minus_HardMTL": (
        "MMoE_streamflow_nse",
        "Hard_MTL_streamflow_nse",
    ),
    "Delta_NSE_CGC_minus_HardMTL": (
        "CGC_streamflow_nse",
        "Hard_MTL_streamflow_nse",
    ),
    "Delta_NSE_CGC_minus_MMoE": (
        "CGC_streamflow_nse",
        "MMoE_streamflow_nse",
    ),

    # Evapotranspiration gains relative to STL-ET.
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
    "Delta_NSE_MMoE_ET_minus_HardMTL": (
        "MMoE_evapotranspiration_nse",
        "Hard_MTL_evapotranspiration_nse",
    ),
    "Delta_NSE_CGC_ET_minus_HardMTL": (
        "CGC_evapotranspiration_nse",
        "Hard_MTL_evapotranspiration_nse",
    ),
    "Delta_NSE_CGC_ET_minus_MMoE": (
        "CGC_evapotranspiration_nse",
        "MMoE_evapotranspiration_nse",
    ),
}


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize CAMELS gauge ids as 8-digit strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def read_summary(model_name: str, model_dir: Path) -> Dict[str, object]:
    """Read validation summary for one model."""
    path = model_dir / "validation_summary.csv"

    record: Dict[str, object] = {
        "Model": model_name,
        "Status": "missing",
        "Run_Dir": str(model_dir),
    }

    if not path.exists():
        return record

    df = pd.read_csv(path)
    if df.empty:
        record["Status"] = "empty"
        return record

    record.update(df.iloc[0].to_dict())
    record["Status"] = "completed"
    return record


def read_per_basin(model_name: str, model_dir: Path) -> pd.DataFrame:
    """Read per-basin validation metrics for one model."""
    path = model_dir / "validation_per_basin_metrics.csv"

    if not path.exists():
        print(f"[Warning] Missing per-basin metrics for {model_name}: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        print(f"[Warning] Empty per-basin metrics for {model_name}: {path}")
        return pd.DataFrame()

    if "gauge_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "gauge_id"})

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])

    rename_map = {
        col: f"{model_name}_{col}"
        for col in df.columns
        if col != "gauge_id"
    }

    return df.rename(columns=rename_map)


def merge_per_basin_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge all per-basin metric tables by gauge id."""
    valid_tables = [table for table in tables if not table.empty]

    if not valid_tables:
        return pd.DataFrame()

    merged = valid_tables[0]

    for table in valid_tables[1:]:
        merged = merged.merge(table, on="gauge_id", how="outer")

    return merged


def add_gain_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standardized NSE gain columns for streamflow and ET."""
    out = df.copy()

    for new_col, (model_col, baseline_col) in GAIN_DEFINITIONS.items():
        if model_col not in out.columns or baseline_col not in out.columns:
            print(
                f"[Warning] Cannot compute {new_col}: "
                f"missing '{model_col}' or '{baseline_col}'."
            )
            continue

        out[new_col] = (
            pd.to_numeric(out[model_col], errors="coerce")
            - pd.to_numeric(out[baseline_col], errors="coerce")
        )

    return out


def add_gain_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add positive, negative, and neutral gain flags."""
    out = df.copy()

    for col in GAIN_DEFINITIONS:
        if col not in out.columns:
            continue

        values = pd.to_numeric(out[col], errors="coerce")
        flag_col = col.replace("Delta_NSE_", "Gain_Flag_")

        out[flag_col] = np.select(
            [values > 0.0, values < 0.0],
            ["positive", "negative"],
            default="neutral",
        )

    return out


def validate_output(df: pd.DataFrame) -> None:
    """Print a compact validation report for required output columns."""
    validation_groups = {
        "Required metric columns": [
            "STL_Q_streamflow_nse",
            "STL_ET_evapotranspiration_nse",
            "Hard_MTL_streamflow_nse",
            "Hard_MTL_evapotranspiration_nse",
            "MMoE_streamflow_nse",
            "MMoE_evapotranspiration_nse",
            "CGC_streamflow_nse",
            "CGC_evapotranspiration_nse",
        ],
        "Streamflow gain columns": [
            "Delta_NSE_HardMTL_minus_STLQ",
            "Delta_NSE_MMoE_minus_STLQ",
            "Delta_NSE_CGC_minus_STLQ",
            "Delta_NSE_MMoE_minus_HardMTL",
            "Delta_NSE_CGC_minus_HardMTL",
            "Delta_NSE_CGC_minus_MMoE",
        ],
        "Evapotranspiration gain columns": [
            "Delta_NSE_HardMTL_ET_minus_STLET",
            "Delta_NSE_MMoE_ET_minus_STLET",
            "Delta_NSE_CGC_ET_minus_STLET",
            "Delta_NSE_MMoE_ET_minus_HardMTL",
            "Delta_NSE_CGC_ET_minus_HardMTL",
            "Delta_NSE_CGC_ET_minus_MMoE",
        ],
        "Gain flag columns": [
            "Gain_Flag_HardMTL_minus_STLQ",
            "Gain_Flag_MMoE_minus_STLQ",
            "Gain_Flag_CGC_minus_STLQ",
            "Gain_Flag_CGC_minus_HardMTL",
            "Gain_Flag_CGC_minus_MMoE",
            "Gain_Flag_HardMTL_ET_minus_STLET",
            "Gain_Flag_MMoE_ET_minus_STLET",
            "Gain_Flag_CGC_ET_minus_STLET",
            "Gain_Flag_CGC_ET_minus_HardMTL",
            "Gain_Flag_CGC_ET_minus_MMoE",
        ],
    }

    print("=" * 100)
    print("Chapter 3 summary validation")
    print(f"Number of basins: {len(df)}")

    all_missing = []

    for group_name, columns in validation_groups.items():
        missing = [col for col in columns if col not in df.columns]
        passed = len(columns) - len(missing)

        print(f"{group_name}: {passed}/{len(columns)} passed")

        if missing:
            all_missing.extend(missing)
            for col in missing:
                print(f"  Missing: {col}")

    if all_missing:
        raise ValueError(
            "Chapter 3 summary validation failed. "
            f"Missing columns: {all_missing}"
        )

    print("Validation passed.")
    print("=" * 100)


def main() -> None:
    """Summarize Chapter 3 formal model results."""
    summary_records = []
    per_basin_tables = []

    for model_name, model_dir in MODEL_DIRS.items():
        summary_records.append(read_summary(model_name, model_dir))
        per_basin_tables.append(read_per_basin(model_name, model_dir))

    summary_df = pd.DataFrame(summary_records)
    summary_path = SUMMARY_DIR / "ch3_performance_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    per_basin_df = merge_per_basin_tables(per_basin_tables)
    if per_basin_df.empty:
        raise ValueError("No Chapter 3 per-basin metric tables were found.")

    per_basin_df = add_gain_columns(per_basin_df)
    per_basin_df = add_gain_flags(per_basin_df)

    per_basin_path = SUMMARY_DIR / "ch3_per_basin_all_models.csv"
    per_basin_df.to_csv(per_basin_path, index=False)

    validate_output(per_basin_df)

    print(f"Saved: {summary_path}")
    print(f"Saved: {per_basin_path}")


if __name__ == "__main__":
    main()