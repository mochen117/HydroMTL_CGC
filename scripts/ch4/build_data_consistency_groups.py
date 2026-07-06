# ==============================================================================
# Description:
#   Build basin groups for Chapter 4 climate-consistency experiments.
#
# Purpose:
#   Quantify train-test climate forcing consistency using joint P-T-CAPE
#   climatic feature distance. Each basin is represented by:
#       [P_mean, P_std, T_mean, T_std, CAPE_mean, CAPE_std]
#   for both training and testing periods. Smaller distance indicates higher
#   climate consistency between training and testing conditions.
#
# Hydrological interpretation:
#   - Precipitation represents water input.
#   - Temperature represents thermal background and snow-related controls.
#   - CAPE represents atmospheric convective instability and storm potential.
#
# Outputs:
#   - experiments/formal_ch4_training_experiments/summary/ch4_climate_consistency_groups.csv
#   - experiments/formal_ch4_training_experiments/basin_groups/consistency_low.txt
#   - experiments/formal_ch4_training_experiments/basin_groups/consistency_medium.txt
#   - experiments/formal_ch4_training_experiments/basin_groups/consistency_high.txt
# ==============================================================================

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "mtl_cgc" / "configs" / "default.yaml"

CH4_DIR = PROJECT_ROOT / "experiments" / "formal_ch4_training_experiments"
SUMMARY_DIR = CH4_DIR / "summary"
GROUP_DIR = CH4_DIR / "basin_groups"

OUTPUT_PATH = SUMMARY_DIR / "ch4_climate_consistency_groups.csv"

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
    "cape": [
        "potential_energy",
        "cape",
        "convective_available_potential_energy",
        "CAPE",
    ],
}

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
GROUP_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_gauge_id(value: object) -> str:
    """Normalize basin id as an 8-digit string."""
    return str(value).strip().replace(".0", "").zfill(8)


def load_config(path: Path) -> dict:
    """Load YAML configuration."""
    require_file(path)
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def discover_basin_files(data_root: Path) -> List[Path]:
    """Discover basin NetCDF files."""
    files = sorted(data_root.glob("gage_*.nc"))
    if not files:
        raise FileNotFoundError(f"No gage_*.nc files found in: {data_root}")
    return files


def infer_variable(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    """Infer variable name from candidate names."""
    lower_map = {name.lower(): name for name in ds.data_vars}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def infer_time_dim(da: xr.DataArray) -> str:
    """Infer time dimension name."""
    for dim in da.dims:
        if dim.lower() in {"time", "date", "datetime"}:
            return dim
    return da.dims[0]


def compute_period_stats(da: xr.DataArray, period: List[str]) -> Tuple[float, float]:
    """Compute mean and standard deviation during one period."""
    time_dim = infer_time_dim(da)
    sub = da.sel({time_dim: slice(period[0], period[1])})

    values = np.asarray(sub.values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    return float(np.mean(values)), float(np.std(values))


def compute_basin_features(
    nc_path: Path,
    train_period: List[str],
    test_period: List[str],
) -> Dict[str, object]:
    """Compute train-test P-T-CAPE climate features for one basin."""
    gauge_id = normalize_gauge_id(nc_path.stem.replace("gage_", ""))

    with xr.open_dataset(nc_path) as ds:
        p_var = infer_variable(ds, VARIABLE_CANDIDATES["precipitation"])
        t_var = infer_variable(ds, VARIABLE_CANDIDATES["temperature"])
        cape_var = infer_variable(ds, VARIABLE_CANDIDATES["cape"])

        if p_var is None or t_var is None or cape_var is None:
            missing = []
            if p_var is None:
                missing.append("precipitation")
            if t_var is None:
                missing.append("temperature")
            if cape_var is None:
                missing.append("CAPE")

            raise ValueError(
                f"Missing required climate variables in {nc_path.name}: {missing}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        p_train_mean, p_train_std = compute_period_stats(ds[p_var], train_period)
        t_train_mean, t_train_std = compute_period_stats(ds[t_var], train_period)
        cape_train_mean, cape_train_std = compute_period_stats(ds[cape_var], train_period)

        p_test_mean, p_test_std = compute_period_stats(ds[p_var], test_period)
        t_test_mean, t_test_std = compute_period_stats(ds[t_var], test_period)
        cape_test_mean, cape_test_std = compute_period_stats(ds[cape_var], test_period)

    return {
        "gauge_id": gauge_id,
        "precipitation_variable": p_var,
        "temperature_variable": t_var,
        "cape_variable": cape_var,
        "p_mean_train": p_train_mean,
        "p_std_train": p_train_std,
        "t_mean_train": t_train_mean,
        "t_std_train": t_train_std,
        "cape_mean_train": cape_train_mean,
        "cape_std_train": cape_train_std,
        "p_mean_test": p_test_mean,
        "p_std_test": p_test_std,
        "t_mean_test": t_test_mean,
        "t_std_test": t_test_std,
        "cape_mean_test": cape_test_mean,
        "cape_std_test": cape_test_std,
    }


def compute_climate_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standardized joint P-T-CAPE train-test climate distance."""
    out = df.copy()

    train_cols = [
        "p_mean_train",
        "p_std_train",
        "t_mean_train",
        "t_std_train",
        "cape_mean_train",
        "cape_std_train",
    ]

    test_cols = [
        "p_mean_test",
        "p_std_test",
        "t_mean_test",
        "t_std_test",
        "cape_mean_test",
        "cape_std_test",
    ]

    feature_df = out[train_cols + test_cols].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    if feature_df.isna().any().any():
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)

    n_features = len(train_cols)
    train_scaled = scaled[:, :n_features]
    test_scaled = scaled[:, n_features:]

    distance = np.linalg.norm(train_scaled - test_scaled, axis=1)

    out["climate_distance"] = distance
    out["climate_similarity"] = 1.0 / (1.0 + distance)

    out["consistency_group"] = pd.qcut(
        out["climate_similarity"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )

    return out.sort_values("climate_similarity", ascending=False)


def write_group_files(df: pd.DataFrame) -> None:
    """Write one basin-list file per climate-consistency group."""
    for group in ["low", "medium", "high"]:
        sub = df[df["consistency_group"].astype(str) == group].copy()
        basin_ids = sub["gauge_id"].tolist()

        path = GROUP_DIR / f"consistency_{group}.txt"
        path.write_text("\n".join(basin_ids) + "\n", encoding="utf-8")

        print(f"Saved: {path} ({len(basin_ids)} basins)")


def main() -> None:
    """Build P-T-CAPE climate-consistency basin groups."""
    cfg = load_config(BASE_CONFIG)

    data_root = Path(cfg["data"]["data_root"])
    train_period = cfg["data"]["train_period"]
    test_period = cfg["data"]["test_period"]

    basin_files = discover_basin_files(data_root)
    records = []

    for idx, nc_path in enumerate(basin_files, start=1):
        if idx == 1 or idx % 50 == 0:
            print(f"Processing basin {idx}/{len(basin_files)}")
        records.append(
            compute_basin_features(
                nc_path=nc_path,
                train_period=train_period,
                test_period=test_period,
            )
        )

    feature_df = pd.DataFrame(records)
    output_df = compute_climate_similarity(feature_df)

    output_df.to_csv(OUTPUT_PATH, index=False)
    write_group_files(output_df)

    print(f"Saved: {OUTPUT_PATH}")
    print(
        "Definition: higher climate similarity means smaller train-test "
        "P-T-CAPE feature distance."
    )


if __name__ == "__main__":
    main()