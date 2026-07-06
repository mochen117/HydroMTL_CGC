# ==============================================================================
# Description:
#   Build a complete CAMELS-US basin metadata table for HydroMTL_CGC.
#
# Purpose:
#   Merge CAMELS name, topographic, climatic, vegetation, and soil attributes
#   into one standardized metadata table. The output is used for Chapter 3 and
#   Chapter 4 post-analysis, including HUC2 grouping, spatial maps, aridity
#   grouping, snow grouping, vegetation grouping, soil grouping, and topographic
#   grouping.
#
# Inputs:
#   - camels_name.txt
#   - camels_topo.txt
#   - camels_clim.txt
#   - camels_vege.txt
#   - camels_soil.txt
#
# Output:
#   - output_592_basins/basin_metadata.csv
# ==============================================================================

from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path("/home/mochen/code/HydroMTL_CGC")
CAMELS_ROOT = Path("/home/mochen/hydro_data/camels/camels_us")
OUTPUT_PATH = PROJECT_ROOT / "output_592_basins" / "basin_metadata.csv"

CAMELS_FILES: Dict[str, Path] = {
    "name": CAMELS_ROOT / "camels_name.txt",
    "topo": CAMELS_ROOT / "camels_topo.txt",
    "clim": CAMELS_ROOT / "camels_clim.txt",
    "vege": CAMELS_ROOT / "camels_vege.txt",
    "soil": CAMELS_ROOT / "camels_soil.txt",
}


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize CAMELS gauge ids as 8-character strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(8)
    )


def read_camels_attribute_file(path: Path) -> pd.DataFrame:
    """Read one CAMELS semicolon-separated attribute file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CAMELS attribute file: {path}")

    df = pd.read_csv(path, sep=";", dtype={"gauge_id": str})

    if "gauge_id" not in df.columns:
        raise ValueError(f"'gauge_id' column not found in {path}")

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])
    return df


def merge_attribute_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge CAMELS attribute tables by gauge id."""
    merged = None

    for name, df in tables.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df,
                on="gauge_id",
                how="outer",
                suffixes=("", f"_{name}"),
            )

    if merged is None:
        raise ValueError("No CAMELS attribute tables were loaded.")

    return merged


def standardize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and retain analysis-ready attributes."""
    out = df.copy()

    out["basin_id"] = normalize_gauge_id(out["gauge_id"])

    if "gauge_lat" in out.columns:
        out["latitude"] = pd.to_numeric(out["gauge_lat"], errors="coerce")

    if "gauge_lon" in out.columns:
        out["longitude"] = pd.to_numeric(out["gauge_lon"], errors="coerce")

    if "aridity" in out.columns:
        out["aridity_index"] = pd.to_numeric(out["aridity"], errors="coerce")

    if "frac_snow" in out.columns:
        out["snow_fraction"] = pd.to_numeric(out["frac_snow"], errors="coerce")

    if "huc_02" in out.columns:
        out["huc_02"] = (
            out["huc_02"]
            .astype(str)
            .str.strip()
            .str.replace(".0", "", regex=False)
            .str.zfill(2)
        )

    ordered_columns: List[str] = [
        "basin_id",
        "gauge_id",
        "huc_02",
        "gauge_name",
        "latitude",
        "longitude",
        "gauge_lat",
        "gauge_lon",
        "elev_mean",
        "slope_mean",
        "area_gages2",
        "area_geospa_fabric",
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

    keep = [col for col in ordered_columns if col in out.columns]
    out = out[keep].drop_duplicates(subset=["basin_id"]).sort_values("basin_id")

    return out


def main() -> None:
    """Build and save the complete CAMELS-US metadata table."""
    print("\n" + "=" * 100)
    print("Build complete CAMELS-US basin metadata")
    print("-" * 100)

    tables: Dict[str, pd.DataFrame] = {}

    for name, path in CAMELS_FILES.items():
        print(f"Reading {name:>5}: {path}")
        tables[name] = read_camels_attribute_file(path)

    metadata = merge_attribute_tables(tables)
    metadata = standardize_metadata(metadata)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(OUTPUT_PATH, index=False)

    print("-" * 100)
    print(f"Saved metadata : {OUTPUT_PATH}")
    print(f"Rows           : {len(metadata)}")
    print(f"Columns        : {len(metadata.columns)}")
    print(f"Columns list   : {list(metadata.columns)}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()