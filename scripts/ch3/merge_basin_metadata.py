# ==============================================================================
# Description:
#   Merge Chapter 3 per-basin model metrics with CAMELS basin metadata.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_all_models.csv
#   - data/basin_metadata.csv
#
# Output:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_with_metadata.csv
# ==============================================================================

from pathlib import Path
from typing import List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling"
SUMMARY_DIR = CH3_DIR / "06_summary"

METRICS_PATH = SUMMARY_DIR / "ch3_per_basin_all_models.csv"
OUTPUT_PATH = SUMMARY_DIR / "ch3_per_basin_with_metadata.csv"

METADATA_CANDIDATES = [
    PROJECT_ROOT / "output_592_basins" / "basin_metadata.csv",
    PROJECT_ROOT / "data" / "basin_metadata.csv",
    PROJECT_ROOT / "data" / "CAMELS_US_custom" / "basin_metadata.csv",
    PROJECT_ROOT / "mtl_cgc" / "data" / "basin_metadata.csv",
]

METADATA_COLUMNS = [
    "gauge_id",
    "huc_02",
    "gauge_name",
    "latitude",
    "longitude",
    "gauge_lat",
    "gauge_lon",
    "area_gages2",
    "area_geospa_fabric",
    "elev_mean",
    "slope_mean",
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "frac_snow",
    "snow_fraction",
    "aridity",
    "aridity_index",
    "high_prec_freq",
    "high_prec_dur",
    "high_prec_timing",
    "low_prec_freq",
    "low_prec_dur",
    "low_prec_timing",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "root_depth_99",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "water_frac",
    "organic_frac",
    "other_frac",
]


def find_metadata_path() -> Path:
    """Find the basin metadata file."""
    for path in METADATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Basin metadata file was not found.")


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize CAMELS gauge id as an 8-digit string."""
    return series.astype(str).str.replace(".0", "", regex=False).str.zfill(8)


def infer_gauge_column(df: pd.DataFrame) -> Optional[str]:
    """Infer gauge id column name from common CAMELS variants."""
    candidates = ["gauge_id", "gage_id", "basin_id", "GAGE_ID"]
    lower_map = {col.lower(): col for col in df.columns}

    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def select_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Select columns that exist in the dataframe."""
    return [col for col in columns if col in df.columns]


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    metadata_path = find_metadata_path()

    metrics = pd.read_csv(METRICS_PATH, dtype={"gauge_id": str})
    metadata = pd.read_csv(
        metadata_path,
        dtype={
            "gauge_id": str,
            "basin_id": str,
            "huc_02": str,
        },
    )

    metrics_gauge_col = infer_gauge_column(metrics)
    metadata_gauge_col = infer_gauge_column(metadata)

    if metrics_gauge_col is None:
        raise ValueError("Metrics table does not contain a gauge id column.")
    if metadata_gauge_col is None:
        raise ValueError("Metadata table does not contain a gauge id column.")

    metrics = metrics.rename(columns={metrics_gauge_col: "gauge_id"})
    metadata = metadata.rename(columns={metadata_gauge_col: "gauge_id"})

    metrics["gauge_id"] = normalize_gauge_id(metrics["gauge_id"])
    metadata["gauge_id"] = normalize_gauge_id(metadata["gauge_id"])
    if "huc_02" in metadata.columns:
        metadata["huc_02"] = (
            metadata["huc_02"]
            .astype(str)
            .str.strip()
            .str.replace(".0", "", regex=False)
            .str.zfill(2)
        )

    keep_cols = select_existing_columns(metadata, METADATA_COLUMNS)
    if "gauge_id" not in keep_cols:
        keep_cols = ["gauge_id"] + keep_cols
    metadata = metadata[keep_cols]

    duplicated_count = metadata["gauge_id"].duplicated().sum()
    if duplicated_count > 0:
        print(f"[Warning] Found {duplicated_count} duplicated gauge_id rows in metadata. Keeping first occurrence.")

    metadata = metadata.drop_duplicates(subset=["gauge_id"])

    merged = metrics.merge(metadata, on="gauge_id", how="left")
    if "aridity" not in merged.columns and "aridity_index" in merged.columns:
        merged["aridity"] = merged["aridity_index"]

    if "frac_snow" not in merged.columns and "snow_fraction" in merged.columns:
        merged["frac_snow"] = merged["snow_fraction"]

    metadata_check_cols = [col for col in ["huc_02", "aridity", "frac_snow", "aridity_index", "snow_fraction"] if col in merged.columns]

    if metadata_check_cols:
        missing_metadata = merged[metadata_check_cols].isna().all(axis=1).sum()
        if missing_metadata > 0:
            print(f"[Warning] Missing metadata for {missing_metadata} gauges.")
    else:
        print("[Warning] No recognized metadata attributes were merged.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Metadata used: {metadata_path}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()