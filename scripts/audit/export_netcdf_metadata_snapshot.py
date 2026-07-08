"""Export lightweight NetCDF metadata for HydroMTL data auditing.

This script scans a small number of processed basin NetCDF files and exports
variable attributes, value ranges, missing ratios, and valid-time intervals.
It is read-only and does not modify any data files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


DATA_ROOT = Path("/home/mochen/code/HydroMTL_CGC/output_592_basins")
OUT_PATH = Path("experiments/audit_units/netcdf_metadata_snapshot.json")
MAX_FILES = 10


def summarize_array(values: np.ndarray) -> dict[str, Any]:
    """Return robust summary statistics for a numeric array."""
    arr = np.asarray(values)

    if not np.issubdtype(arr.dtype, np.number):
        return {
            "dtype": str(arr.dtype),
            "finite_count": None,
            "nan_ratio": None,
            "min": None,
            "p01": None,
            "median": None,
            "mean": None,
            "p99": None,
            "max": None,
        }

    arr = arr.astype(float).ravel()
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return {
            "dtype": str(values.dtype),
            "finite_count": 0,
            "nan_ratio": float(np.isnan(arr).mean()) if arr.size else None,
            "min": None,
            "p01": None,
            "median": None,
            "mean": None,
            "p99": None,
            "max": None,
        }

    return {
        "dtype": str(values.dtype),
        "finite_count": int(finite.size),
        "nan_ratio": float(np.isnan(arr).mean()) if arr.size else 0.0,
        "min": float(np.nanmin(finite)),
        "p01": float(np.nanpercentile(finite, 1)),
        "median": float(np.nanmedian(finite)),
        "mean": float(np.nanmean(finite)),
        "p99": float(np.nanpercentile(finite, 99)),
        "max": float(np.nanmax(finite)),
    }


def summarize_valid_time(ds: xr.Dataset, var_name: str) -> dict[str, Any]:
    """Summarize valid observation dates for a time-dependent variable."""
    if var_name not in ds:
        return {}

    var = ds[var_name]

    if "time" not in ds.coords or "time" not in var.dims:
        return {
            "is_time_dependent": False,
        }

    time = pd.to_datetime(np.asarray(ds["time"].values))
    values = np.asarray(var.values)

    time_axis = var.dims.index("time")
    values_by_time = np.moveaxis(values, time_axis, 0).reshape(len(time), -1)
    valid = np.isfinite(values_by_time).any(axis=1)

    valid_time = np.asarray(time)[valid]

    result: dict[str, Any] = {
        "is_time_dependent": True,
        "valid_count": int(valid_time.size),
        "first_valid": str(valid_time[0]) if valid_time.size else None,
        "last_valid": str(valid_time[-1]) if valid_time.size else None,
    }

    if valid_time.size >= 2:
        intervals = np.diff(valid_time).astype("timedelta64[D]").astype(int)
        unique_intervals = sorted({int(x) for x in intervals.ravel().tolist()})

        result.update(
            {
                "median_valid_interval_days": float(np.median(intervals)),
                "max_valid_interval_days": int(np.max(intervals)),
                "unique_valid_interval_days_head": unique_intervals[:20],
            }
        )

    return result


def json_safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Convert xarray attributes into JSON-serializable objects."""
    safe: dict[str, Any] = {}

    for key, value in attrs.items():
        if isinstance(value, bytes):
            safe[key] = value.decode("utf-8", errors="replace")
        elif isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        else:
            safe[key] = value

    return safe


def main() -> None:
    """Scan NetCDF files and export lightweight metadata."""
    nc_files = sorted(DATA_ROOT.rglob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found under {DATA_ROOT}")

    records: list[dict[str, Any]] = []

    for path in nc_files[:MAX_FILES]:
        with xr.open_dataset(path) as ds:
            record: dict[str, Any] = {
                "file": str(path),
                "dims": {str(k): int(v) for k, v in ds.sizes.items()},
                "coords": list(ds.coords),
                "data_vars": list(ds.data_vars),
                "global_attrs": json_safe_attrs(dict(ds.attrs)),
                "variables": {},
            }

            for var_name in ds.data_vars:
                var = ds[var_name]
                record["variables"][var_name] = {
                    "dims": list(var.dims),
                    "shape": list(var.shape),
                    "attrs": json_safe_attrs(dict(var.attrs)),
                    "summary": summarize_array(var.values),
                    "valid_time": summarize_valid_time(ds, var_name),
                }

            records.append(record)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Metadata written to {OUT_PATH}")


if __name__ == "__main__":
    main()