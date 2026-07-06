# ==============================================================================
# Description:
#   Summarize Chapter 4 controlled training experiments.
#
# Purpose:
#   Collect validation summaries and per-basin metrics from training-length,
#   climate-consistency, and basin-diversity experiments. The script supports
#   CGC multi-task runs, STL-Q streamflow-only baselines, and STL-ET
#   evapotranspiration-only baselines.
#
# Outputs:
#   - ch4_training_experiment_summary.csv
#   - ch4_training_experiment_per_basin.csv
# ==============================================================================

from pathlib import Path
from typing import Dict, List

import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
SUMMARY_DIR = CH4_DIR / "summary"

SUMMARY_PATH = SUMMARY_DIR / "ch4_training_experiment_summary.csv"
PER_BASIN_PATH = SUMMARY_DIR / "ch4_training_experiment_per_basin.csv"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize basin ids as 8-digit strings."""
    return series.astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(8)


def infer_experiment_type(run_name: str) -> str:
    """Infer experiment type from run name."""
    if run_name.startswith("ch4_length"):
        return "training_length"
    if run_name.startswith("ch4_consistency"):
        return "climate_consistency"
    if run_name.startswith("ch4_diversity"):
        return "basin_diversity"
    return "unknown"


def infer_model_name(run_name: str) -> str:
    """Infer model name from run name."""
    name = run_name.lower()

    if "_stlq_" in name:
        return "STL-Q"
    if "_stlet_" in name:
        return "STL-ET"
    if "_cgc_" in name:
        return "CGC"

    return "unknown"


def parse_group_name(run_name: str) -> str:
    """Parse experimental group from run name."""
    parts = run_name.split("_")

    if run_name.startswith("ch4_length"):
        return "_".join(parts[2:4])
    if run_name.startswith("ch4_consistency"):
        return parts[2]
    if run_name.startswith("ch4_diversity"):
        return parts[2]

    return "unknown"


def collect_run_dirs() -> List[Path]:
    """Collect Chapter 4 run directories."""
    if not CH4_DIR.exists():
        return []

    return sorted(
        path
        for path in CH4_DIR.iterdir()
        if path.is_dir() and path.name.startswith("ch4_")
    )


def read_metadata(run_dir: Path) -> Dict[str, object]:
    """Read optional metadata.json for one run."""
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except Exception:
        return {}

    keep_keys = [
        "num_train_basins",
        "num_test_basins",
        "train_period",
        "val_period",
        "test_period",
        "split_label",
        "model_architecture",
    ]

    return {key: metadata.get(key) for key in keep_keys if key in metadata}


def read_validation_summary(run_dir: Path) -> Dict[str, object]:
    """Read validation summary for one run."""
    path = run_dir / "validation_summary.csv"

    record: Dict[str, object] = {
        "run_name": run_dir.name,
        "experiment_type": infer_experiment_type(run_dir.name),
        "group_name": parse_group_name(run_dir.name),
        "model_name": infer_model_name(run_dir.name),
        "status": "missing",
    }

    record.update(read_metadata(run_dir))

    if not path.exists():
        return record

    summary = pd.read_csv(path)
    if summary.empty:
        record["status"] = "empty"
        return record

    record.update(summary.iloc[0].to_dict())
    record["status"] = "completed"

    return record


def read_per_basin_metrics(run_dir: Path) -> pd.DataFrame:
    """Read per-basin validation metrics for one run."""
    path = run_dir / "validation_per_basin_metrics.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"gauge_id": str})
    if df.empty:
        return pd.DataFrame()

    if "gauge_id" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "gauge_id"})

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])
    df.insert(0, "model_name", infer_model_name(run_dir.name))
    df.insert(0, "group_name", parse_group_name(run_dir.name))
    df.insert(0, "experiment_type", infer_experiment_type(run_dir.name))
    df.insert(0, "run_name", run_dir.name)

    return df


def main() -> None:
    """Summarize all Chapter 4 controlled training experiments."""
    run_dirs = collect_run_dirs()

    summary_records = [read_validation_summary(run_dir) for run_dir in run_dirs]
    summary_df = pd.DataFrame(summary_records)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["experiment_type", "group_name", "model_name", "run_name"],
            ignore_index=True,
        )

    summary_df.to_csv(SUMMARY_PATH, index=False)

    per_basin_tables = [read_per_basin_metrics(run_dir) for run_dir in run_dirs]
    per_basin_tables = [df for df in per_basin_tables if not df.empty]

    if per_basin_tables:
        per_basin_df = pd.concat(per_basin_tables, ignore_index=True)
    else:
        per_basin_df = pd.DataFrame()

    per_basin_df.to_csv(PER_BASIN_PATH, index=False)

    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {PER_BASIN_PATH}")
    print(f"Runs found: {len(run_dirs)}")

    if not summary_df.empty:
        completed = int((summary_df["status"] == "completed").sum())
        print(f"Completed runs: {completed}")
        print(summary_df["model_name"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()