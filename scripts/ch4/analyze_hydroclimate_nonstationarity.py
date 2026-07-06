# ==============================================================================
# Copyright (c) 2024-2026. All Rights Reserved.
# Description:
#   Analyze hydroclimatic nonstationarity for Chapter 4.
#
# Purpose:
#   Perform basin-level Mann-Kendall trend tests on annual hydroclimatic
#   variables, including precipitation, temperature, evapotranspiration, and
#   streamflow. The results support the Chapter 4 motivation that training and
#   testing periods may differ under changing hydroclimatic conditions.
#
# Inputs:
#   - NetCDF basin files defined by mtl_cgc/configs/default.yaml
#
# Outputs:
#   - ch4_hydroclimate_nonstationarity_per_basin.csv
#   - ch4_hydroclimate_nonstationarity_summary.csv
#   - ch4_hydroclimate_representative_basins.csv
# ==============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"

CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
SUMMARY_DIR = CH4_DIR / "summary"

PER_BASIN_OUTPUT = SUMMARY_DIR / "ch4_hydroclimate_nonstationarity_per_basin.csv"
SUMMARY_OUTPUT = SUMMARY_DIR / "ch4_hydroclimate_nonstationarity_summary.csv"
REPRESENTATIVE_OUTPUT = SUMMARY_DIR / "ch4_hydroclimate_representative_basins.csv"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

SIGNIFICANCE_LEVEL = 0.10
Z_CRITICAL = float(norm.ppf(1.0 - SIGNIFICANCE_LEVEL / 2.0))

VARIABLE_CANDIDATES: Dict[str, List[str]] = {
    "precipitation": [
        "total_precipitation",
        "precipitation",
        "precip",
        "prcp",
        "P",
    ],
    "temperature": [
        "temperature",
        "temp",
        "t_mean",
        "tas",
        "T",
    ],
    "evapotranspiration": [
        "evapotranspiration",
        "actual_evapotranspiration",
        "ET",
        "aet",
    ],
    "streamflow": [
        "streamflow",
        "discharge",
        "runoff",
        "Q",
    ],
}


def require_file(path: Path) -> None:
    """Validate that a required file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_yaml(path: Path) -> dict:
    """Load YAML configuration."""
    require_file(path)
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_gauge_id(value: object) -> str:
    """Normalize basin ID as an 8-digit string."""
    return str(value).strip().replace(".0", "").zfill(8)


def discover_basin_files(data_root: Path) -> List[Path]:
    """Discover basin NetCDF files."""
    files = sorted(data_root.glob("gage_*.nc"))
    if not files:
        raise FileNotFoundError(f"No gage_*.nc files found in: {data_root}")
    return files


def infer_variable(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    """Infer a variable name from candidate names."""
    lower_map = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def infer_time_dim(da: xr.DataArray) -> str:
    """Infer the time dimension of a DataArray."""
    for dim in da.dims:
        if dim.lower() in {"time", "date", "datetime"}:
            return dim
    return da.dims[0]


def annual_series(da: xr.DataArray, period: List[str]) -> pd.Series:
    """Aggregate a daily or monthly series to annual values."""
    time_dim = infer_time_dim(da)
    sub = da.sel({time_dim: slice(period[0], period[1])})

    if sub.size == 0:
        return pd.Series(dtype=float)

    frame = sub.to_dataframe(name="value").reset_index()
    time_col = time_dim

    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=[time_col, "value"])

    if frame.empty:
        return pd.Series(dtype=float)

    frame["year"] = frame[time_col].dt.year

    annual = frame.groupby("year")["value"].mean()
    annual = annual.replace([np.inf, -np.inf], np.nan).dropna()

    return annual


def mann_kendall_test(values: pd.Series) -> Dict[str, float]:
    """Run a two-sided Mann-Kendall trend test."""
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)

    if n < 5:
        return {
            "n_years": n,
            "mk_s": np.nan,
            "mk_var_s": np.nan,
            "mk_z": np.nan,
            "mk_p_value": np.nan,
            "sen_slope": np.nan,
            "trend_direction": "insufficient",
            "significant": False,
        }

    s = 0.0
    for i in range(n - 1):
        s += np.sum(np.sign(x[i + 1:] - x[i]))

    unique_x, counts = np.unique(x, return_counts=True)

    var_s = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(counts * (counts - 1) * (2 * counts + 5))
    ) / 18.0

    if var_s <= 0:
        z_value = 0.0
    elif s > 0:
        z_value = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z_value = (s + 1.0) / np.sqrt(var_s)
    else:
        z_value = 0.0

    p_value = 2.0 * (1.0 - norm.cdf(abs(z_value)))

    slopes = []
    years = values.dropna().index.to_numpy(dtype=float)

    for i in range(n - 1):
        denom = years[i + 1:] - years[i]
        valid = denom != 0
        slopes.extend(((x[i + 1:][valid] - x[i]) / denom[valid]).tolist())

    sen_slope = float(np.median(slopes)) if slopes else np.nan

    significant = abs(z_value) > Z_CRITICAL

    if significant and z_value > 0:
        direction = "increasing"
    elif significant and z_value < 0:
        direction = "decreasing"
    else:
        direction = "not significant"

    return {
        "n_years": n,
        "mk_s": float(s),
        "mk_var_s": float(var_s),
        "mk_z": float(z_value),
        "mk_p_value": float(p_value),
        "sen_slope": sen_slope,
        "trend_direction": direction,
        "significant": bool(significant),
    }


def analyze_one_basin(
    nc_path: Path,
    analysis_period: List[str],
) -> List[Dict[str, object]]:
    """Analyze all target variables for one basin."""
    gauge_id = normalize_gauge_id(nc_path.stem.replace("gage_", ""))
    records: List[Dict[str, object]] = []

    with xr.open_dataset(nc_path) as ds:
        for variable, candidates in VARIABLE_CANDIDATES.items():
            var_name = infer_variable(ds, candidates)

            if var_name is None:
                records.append(
                    {
                        "gauge_id": gauge_id,
                        "variable": variable,
                        "source_variable": np.nan,
                        "n_years": 0,
                        "mk_s": np.nan,
                        "mk_var_s": np.nan,
                        "mk_z": np.nan,
                        "mk_p_value": np.nan,
                        "sen_slope": np.nan,
                        "trend_direction": "missing variable",
                        "significant": False,
                    }
                )
                continue

            series = annual_series(ds[var_name], analysis_period)
            result = mann_kendall_test(series)

            records.append(
                {
                    "gauge_id": gauge_id,
                    "variable": variable,
                    "source_variable": var_name,
                    **result,
                }
            )

    return records


def summarize_results(per_basin: pd.DataFrame) -> pd.DataFrame:
    """Summarize trend-test results by variable."""
    records = []

    for variable, group in per_basin.groupby("variable"):
        valid = group[group["trend_direction"] != "missing variable"].copy()
        n_basins = len(valid)

        increasing = int((valid["trend_direction"] == "increasing").sum())
        decreasing = int((valid["trend_direction"] == "decreasing").sum())
        significant = int(valid["significant"].sum())
        non_significant = int((valid["trend_direction"] == "not significant").sum())

        records.append(
            {
                "variable": variable,
                "n_basins": n_basins,
                "significant_increase_count": increasing,
                "significant_decrease_count": decreasing,
                "significant_count": significant,
                "non_significant_count": non_significant,
                "significant_rate_pct": (
                    significant / n_basins * 100.0 if n_basins > 0 else np.nan
                ),
                "z_critical_alpha_0.10": Z_CRITICAL,
            }
        )

    return pd.DataFrame(records)


def select_representative_basins(per_basin: pd.DataFrame) -> pd.DataFrame:
    """Select representative basins with strongest positive and negative trends."""
    records = []

    for variable, group in per_basin.groupby("variable"):
        valid = group.dropna(subset=["mk_z"]).copy()
        valid = valid[valid["trend_direction"].isin(["increasing", "decreasing"])]

        if valid.empty:
            continue

        strongest_increase = valid.sort_values("mk_z", ascending=False).head(1)
        strongest_decrease = valid.sort_values("mk_z", ascending=True).head(1)

        for label, sub in [
            ("strongest_increase", strongest_increase),
            ("strongest_decrease", strongest_decrease),
        ]:
            if sub.empty:
                continue
            row = sub.iloc[0]
            records.append(
                {
                    "variable": variable,
                    "representative_type": label,
                    "gauge_id": row["gauge_id"],
                    "mk_z": row["mk_z"],
                    "mk_p_value": row["mk_p_value"],
                    "sen_slope": row["sen_slope"],
                    "trend_direction": row["trend_direction"],
                }
            )

    return pd.DataFrame(records)


def main() -> None:
    """Run hydroclimatic nonstationarity analysis."""
    print("=" * 100)
    print("Chapter 4 hydroclimatic nonstationarity analysis")
    print("=" * 100)

    cfg = load_yaml(BASE_CONFIG)
    data_root = Path(cfg["data"]["data_root"])

    train_period = cfg["data"]["train_period"]
    test_period = cfg["data"]["test_period"]
    analysis_period = [train_period[0], test_period[1]]

    basin_files = discover_basin_files(data_root)

    all_records: List[Dict[str, object]] = []

    for idx, nc_path in enumerate(basin_files, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(basin_files):
            print(f"Processing basin {idx}/{len(basin_files)}")
        all_records.extend(analyze_one_basin(nc_path, analysis_period))

    per_basin = pd.DataFrame(all_records)
    summary = summarize_results(per_basin)
    representative = select_representative_basins(per_basin)

    per_basin.to_csv(PER_BASIN_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    representative.to_csv(REPRESENTATIVE_OUTPUT, index=False)

    print(f"Saved: {PER_BASIN_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")
    print(f"Saved: {REPRESENTATIVE_OUTPUT}")
    print("=" * 100)
    print("Hydroclimatic nonstationarity analysis completed.")
    print("=" * 100)


if __name__ == "__main__":
    main()