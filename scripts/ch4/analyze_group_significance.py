# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Test statistical significance of hydrological group differences in CGC
#   performance changes for both streamflow and evapotranspiration.
#
# Purpose:
#   Evaluate whether CGC performance changes relative to single-task baselines
#   differ across hydrological basin groups, including aridity, snow fraction,
#   HUC2 region, elevation, forest cover, and soil porosity.
#
# Inputs:
#   - experiments/formal_ch3_modeling/06_summary/ch3_per_basin_with_metadata.csv
#
# Outputs:
#   - experiments/formal_ch4_training_experiments/summary/
#     ch4_group_significance_tests.csv
# ==============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CH3_SUMMARY_DIR = PROJECT_ROOT / "experiments" / "formal_ch3_modeling" / "06_summary"
CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
CH4_SUMMARY_DIR = CH4_DIR / "summary"

INPUT_PATH = CH3_SUMMARY_DIR / "ch3_per_basin_with_metadata.csv"
OUTPUT_PATH = CH4_SUMMARY_DIR / "ch4_group_significance_tests.csv"

TASK_DELTA_COLUMNS: Dict[str, str] = {
    "streamflow": "Delta_NSE_CGC_minus_STLQ",
    "evapotranspiration": "Delta_NSE_CGC_ET_minus_STLET",
}

MIN_GROUP_SIZE = 5

CH4_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Validate that a required file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_huc2(series: pd.Series) -> pd.Series:
    """Normalize HUC2 region codes as two-digit strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(2)
    )


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first available column from candidate names."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def assign_aridity_group(value: float) -> str | float:
    """Assign aridity groups using common hydrological thresholds."""
    if pd.isna(value):
        return np.nan
    if value < 0.65:
        return "Humid"
    if value < 1.0:
        return "Sub-humid"
    if value < 2.0:
        return "Semi-arid"
    return "Arid"


def add_quantile_group(
    df: pd.DataFrame,
    source_col: str,
    output_col: str,
    labels: List[str],
) -> pd.DataFrame:
    """Add quantile-based group labels safely."""
    out = df.copy()
    values = pd.to_numeric(out[source_col], errors="coerce")

    try:
        grouped = pd.qcut(
            values,
            q=len(labels),
            labels=labels,
            duplicates="drop",
        )
        out[output_col] = grouped
    except ValueError:
        print(f"[Skip] Unable to create quantile group: {output_col}")

    return out


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add hydrological group columns for significance testing."""
    out = df.copy()

    if "huc_02" in out.columns:
        out["huc_02"] = normalize_huc2(out["huc_02"])

    aridity_col = resolve_column(out, ["aridity_index", "aridity"])
    if aridity_col is not None:
        out["aridity_group"] = (
            pd.to_numeric(out[aridity_col], errors="coerce")
            .apply(assign_aridity_group)
        )

    snow_col = resolve_column(out, ["snow_fraction", "frac_snow"])
    if snow_col is not None:
        out = add_quantile_group(
            df=out,
            source_col=snow_col,
            output_col="snow_group",
            labels=[
                "Low snow",
                "Medium-low snow",
                "Medium-high snow",
                "High snow",
            ],
        )

    elev_col = resolve_column(out, ["elev_mean", "elevation_mean"])
    if elev_col is not None:
        out = add_quantile_group(
            df=out,
            source_col=elev_col,
            output_col="elevation_group",
            labels=[
                "Low elevation",
                "Medium-low elevation",
                "Medium-high elevation",
                "High elevation",
            ],
        )

    forest_col = resolve_column(out, ["frac_forest", "forest_fraction"])
    if forest_col is not None:
        out = add_quantile_group(
            df=out,
            source_col=forest_col,
            output_col="forest_group",
            labels=[
                "Low forest",
                "Medium-low forest",
                "Medium-high forest",
                "High forest",
            ],
        )

    porosity_col = resolve_column(out, ["soil_porosity", "porosity"])
    if porosity_col is not None:
        out = add_quantile_group(
            df=out,
            source_col=porosity_col,
            output_col="soil_porosity_group",
            labels=[
                "Low porosity",
                "Medium-low porosity",
                "Medium-high porosity",
                "High porosity",
            ],
        )

    return out


def get_group_values(
    df: pd.DataFrame,
    group_col: str,
    delta_col: str,
) -> Dict[str, pd.Series]:
    """Collect valid Delta NSE values by group."""
    groups: Dict[str, pd.Series] = {}

    for group, sub in df.groupby(group_col, observed=True):
        values = pd.to_numeric(sub[delta_col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) >= MIN_GROUP_SIZE:
            groups[str(group)] = values

    return groups


def run_kruskal_test(
    df: pd.DataFrame,
    task: str,
    delta_col: str,
    group_col: str,
) -> List[Dict[str, object]]:
    """Run Kruskal-Wallis test across all valid groups."""
    groups = get_group_values(df, group_col, delta_col)

    if len(groups) < 2:
        return []

    stat, p_value = kruskal(*groups.values())

    group_medians = {
        name: float(values.median())
        for name, values in groups.items()
    }

    return [
        {
            "task": task,
            "delta_column": delta_col,
            "test_type": "Kruskal-Wallis",
            "group_variable": group_col,
            "group_a": "all",
            "group_b": "all",
            "n_a": int(sum(len(values) for values in groups.values())),
            "n_b": np.nan,
            "statistic": float(stat),
            "p_value": float(p_value),
            "median_a": np.nan,
            "median_b": np.nan,
            "effect_median_difference": np.nan,
            "group_medians": "; ".join(
                f"{name}:{median:.6f}"
                for name, median in group_medians.items()
            ),
        }
    ]


def run_pairwise_mannwhitney(
    df: pd.DataFrame,
    task: str,
    delta_col: str,
    group_col: str,
) -> List[Dict[str, object]]:
    """Run pairwise Mann-Whitney U tests across valid groups."""
    groups = get_group_values(df, group_col, delta_col)
    names = list(groups.keys())
    records: List[Dict[str, object]] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name = names[i]
            b_name = names[j]
            a = groups[a_name]
            b = groups[b_name]

            stat, p_value = mannwhitneyu(a, b, alternative="two-sided")

            median_a = float(a.median())
            median_b = float(b.median())

            records.append(
                {
                    "task": task,
                    "delta_column": delta_col,
                    "test_type": "Mann-Whitney U",
                    "group_variable": group_col,
                    "group_a": a_name,
                    "group_b": b_name,
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "median_a": median_a,
                    "median_b": median_b,
                    "effect_median_difference": median_a - median_b,
                    "group_medians": np.nan,
                }
            )

    return records


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Apply Benjamini-Hochberg false-discovery-rate correction."""
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.dropna()

    out = pd.Series(index=p.index, dtype=float)

    if valid.empty:
        return out

    ranked = valid.sort_values()
    n = len(ranked)

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1]
    adjusted = adjusted.clip(upper=1.0)

    out.loc[adjusted.index] = adjusted
    return out


def validate_delta_columns(df: pd.DataFrame) -> None:
    """Validate that required Delta NSE columns exist."""
    missing = [
        col for col in TASK_DELTA_COLUMNS.values()
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Input table is missing required Delta NSE columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def main() -> None:
    """Run hydrological group significance tests for Q and ET."""
    print("=" * 100)
    print("Chapter 4 hydrological group significance tests")
    print("=" * 100)

    require_file(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, dtype={"gauge_id": str, "huc_02": str})
    validate_delta_columns(df)

    df = add_group_columns(df)

    group_cols = [
        "aridity_group",
        "snow_group",
        "huc_02",
        "elevation_group",
        "forest_group",
        "soil_porosity_group",
    ]

    records: List[Dict[str, object]] = []

    for task, delta_col in TASK_DELTA_COLUMNS.items():
        for group_col in group_cols:
            if group_col not in df.columns:
                print(f"[Skip] Missing group column: {group_col}")
                continue

            records.extend(
                run_kruskal_test(
                    df=df,
                    task=task,
                    delta_col=delta_col,
                    group_col=group_col,
                )
            )
            records.extend(
                run_pairwise_mannwhitney(
                    df=df,
                    task=task,
                    delta_col=delta_col,
                    group_col=group_col,
                )
            )

    result = pd.DataFrame(records)

    if not result.empty:
        result["p_value_fdr"] = benjamini_hochberg(result["p_value"])
        result["significant_p_lt_0.05"] = result["p_value"] < 0.05
        result["significant_fdr_lt_0.05"] = result["p_value_fdr"] < 0.05

        ordered_cols = [
            "task",
            "delta_column",
            "test_type",
            "group_variable",
            "group_a",
            "group_b",
            "n_a",
            "n_b",
            "statistic",
            "p_value",
            "p_value_fdr",
            "significant_p_lt_0.05",
            "significant_fdr_lt_0.05",
            "median_a",
            "median_b",
            "effect_median_difference",
            "group_medians",
        ]
        result = result[ordered_cols]

    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(result)}")
    print("=" * 100)
    print("Hydrological group significance tests completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()