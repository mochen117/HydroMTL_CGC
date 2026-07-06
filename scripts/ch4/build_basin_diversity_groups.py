# ==============================================================================
# Description:
#   Build basin groups for Chapter 4 training-basin diversity experiments.
#
# Purpose:
#   Construct basin subsets with different hydrologic regional diversity levels
#   using HUC2 regions:
#       low    : top 3 HUC2 regions by basin count
#       medium : top 8 HUC2 regions by basin count
#       high   : all available HUC2 regions
#
# Outputs:
#   - ch4_basin_diversity_groups.csv
#   - diversity_low.txt
#   - diversity_medium.txt
#   - diversity_high.txt
# ==============================================================================

from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH3_SUMMARY_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling" / "06_summary"

CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
SUMMARY_DIR = CH4_DIR / "summary"
GROUP_DIR = CH4_DIR / "basin_groups"

INPUT_PATH = CH3_SUMMARY_DIR / "ch3_per_basin_with_metadata.csv"
OUTPUT_PATH = SUMMARY_DIR / "ch4_basin_diversity_groups.csv"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
GROUP_DIR.mkdir(parents=True, exist_ok=True)


def normalize_gauge_id(series: pd.Series) -> pd.Series:
    """Normalize basin ids as 8-digit strings."""
    return series.astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(8)


def normalize_huc2(series: pd.Series) -> pd.Series:
    """Normalize HUC2 codes as 2-digit strings."""
    return series.astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(2)


def load_input() -> pd.DataFrame:
    """Load basin metadata with HUC2 information."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype={"gauge_id": str, "huc_02": str})

    if "gauge_id" not in df.columns:
        raise ValueError("Input table must contain 'gauge_id'.")
    if "huc_02" not in df.columns:
        raise ValueError("Input table must contain 'huc_02'.")

    df["gauge_id"] = normalize_gauge_id(df["gauge_id"])
    df["huc_02"] = normalize_huc2(df["huc_02"])
    df = df.dropna(subset=["gauge_id", "huc_02"]).copy()

    return df


def select_huc_regions(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Select HUC2 regions for low, medium, and high diversity groups."""
    counts = df["huc_02"].value_counts().sort_values(ascending=False)

    low_regions = counts.head(3).index.tolist()
    medium_regions = counts.head(8).index.tolist()
    high_regions = counts.index.tolist()

    return {
        "low": sorted(low_regions),
        "medium": sorted(medium_regions),
        "high": sorted(high_regions),
    }


def assign_groups(df: pd.DataFrame, region_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Create a long-format basin-group table."""
    records = []

    for group_name, huc_list in region_groups.items():
        sub = df[df["huc_02"].isin(huc_list)].copy()
        for _, row in sub.iterrows():
            records.append(
                {
                    "diversity_group": group_name,
                    "gauge_id": row["gauge_id"],
                    "huc_02": row["huc_02"],
                    "n_huc2_regions": len(huc_list),
                }
            )

    return pd.DataFrame(records)


def write_group_files(group_df: pd.DataFrame) -> None:
    """Write one basin-list file per diversity group."""
    for group_name in ["low", "medium", "high"]:
        sub = group_df[group_df["diversity_group"] == group_name].copy()
        path = GROUP_DIR / f"diversity_{group_name}.txt"
        basin_ids = sorted(sub["gauge_id"].unique().tolist())
        path.write_text("\n".join(basin_ids) + "\n", encoding="utf-8")
        print(f"Saved: {path} ({len(basin_ids)} basins)")


def main() -> None:
    """Build basin diversity groups."""
    df = load_input()
    region_groups = select_huc_regions(df)

    print("Selected HUC2 regions:")
    for group_name, regions in region_groups.items():
        print(f"  {group_name:<7}: {regions}")

    group_df = assign_groups(df, region_groups)
    group_df.to_csv(OUTPUT_PATH, index=False)
    write_group_files(group_df)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()