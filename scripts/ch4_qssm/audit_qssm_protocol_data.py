#!/usr/bin/env python3
"""Audit Q-SSM data availability for Chapter 4 experiments.

This script scans basin NetCDF files and reports streamflow/SSM availability,
SSM value range, and time coverage. It does not modify data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xarray is required. Install with: pip install xarray netCDF4") from exc

from ch4_common import STREAMFLOW_ALIASES, SSM_ALIASES, write_basin_file


def _pick_var(ds: xr.Dataset, aliases: Sequence[str]) -> Optional[str]:
    names = list(ds.data_vars) + list(ds.coords)
    lower_to_name = {n.lower(): n for n in names}
    for alias in aliases:
        if alias.lower() in lower_to_name:
            return lower_to_name[alias.lower()]
    # fuzzy contains fallback
    for n in names:
        ln = n.lower()
        for alias in aliases:
            if alias.lower() in ln:
                return n
    return None


def _time_summary(ds: xr.Dataset) -> tuple[Optional[str], Optional[str], Optional[float]]:
    for coord in ("time", "date", "datetime"):
        if coord in ds.coords:
            vals = pd.to_datetime(ds[coord].values)
            if len(vals) == 0:
                return None, None, None
            diffs = np.diff(vals.values.astype("datetime64[D]").astype("int64"))
            median_step = float(np.nanmedian(diffs)) if diffs.size else None
            return str(vals[0].date()), str(vals[-1].date()), median_step
    return None, None, None


def _safe_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype="float64")
    finite = np.isfinite(arr)
    out = {
        "valid_count": int(finite.sum()),
        "total_count": int(arr.size),
        "valid_ratio": float(finite.mean()) if arr.size else 0.0,
        "min": np.nan,
        "p01": np.nan,
        "p50": np.nan,
        "p99": np.nan,
        "max": np.nan,
    }
    if finite.any():
        vals = arr[finite]
        out.update({
            "min": float(np.nanmin(vals)),
            "p01": float(np.nanpercentile(vals, 1)),
            "p50": float(np.nanpercentile(vals, 50)),
            "p99": float(np.nanpercentile(vals, 99)),
            "max": float(np.nanmax(vals)),
        })
    return out


def infer_basin_id(path: Path) -> str:
    stem = path.stem
    for token in stem.replace("-", "_").split("_"):
        if token.isdigit() and 6 <= len(token) <= 12:
            return token.zfill(8)
    return stem


def audit(data_root: Path, out_dir: Path, min_ssm_valid: int = 20, max_files: Optional[int] = None) -> pd.DataFrame:
    files = sorted(data_root.glob("*.nc"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No NetCDF files found directly under {data_root}")

    rows = []
    for i, nc in enumerate(files, 1):
        if i % 100 == 0:
            print(f"Audited {i}/{len(files)} files...")
        row = {"basin_id": infer_basin_id(nc), "file": str(nc)}
        try:
            with xr.open_dataset(nc) as ds:
                q_var = _pick_var(ds, STREAMFLOW_ALIASES)
                ssm_var = _pick_var(ds, SSM_ALIASES)
                t0, t1, dt_med = _time_summary(ds)
                row.update({"time_start": t0, "time_end": t1, "median_time_step_days": dt_med})
                row["q_var"] = q_var
                row["ssm_var"] = ssm_var
                if q_var is not None:
                    stats = _safe_stats(ds[q_var].values)
                    row.update({f"q_{k}": v for k, v in stats.items()})
                else:
                    row.update({"q_valid_count": 0, "q_valid_ratio": 0.0})
                if ssm_var is not None:
                    stats = _safe_stats(ds[ssm_var].values)
                    row.update({f"ssm_{k}": v for k, v in stats.items()})
                else:
                    row.update({"ssm_valid_count": 0, "ssm_valid_ratio": 0.0})
        except Exception as exc:
            row["error"] = repr(exc)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["eligible_qssm"] = (df.get("q_valid_count", 0).fillna(0) > 0) & (df.get("ssm_valid_count", 0).fillna(0) >= min_ssm_valid)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "qssm_data_audit.csv", index=False)
    eligible = df.loc[df["eligible_qssm"], "basin_id"].astype(str).tolist()
    write_basin_file(eligible, out_dir / "eligible_qssm_basins.txt")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit streamflow and SSM data availability.")
    parser.add_argument("--data-root", required=True, type=Path, help="Directory containing one NetCDF file per basin.")
    parser.add_argument("--out-dir", default=Path("experiments/ch4_qssm/audit"), type=Path)
    parser.add_argument("--min-ssm-valid", default=20, type=int)
    parser.add_argument("--max-files", default=None, type=int)
    args = parser.parse_args()
    df = audit(args.data_root, args.out_dir, args.min_ssm_valid, args.max_files)
    print("\nAudit finished.")
    print(f"Basins total: {len(df)}")
    print(f"Eligible Q-SSM basins: {int(df['eligible_qssm'].sum())}")
    if "ssm_p50" in df:
        vals = df["ssm_p50"].dropna()
        if len(vals):
            print(f"SSM median range across basins: {vals.min():.4g} to {vals.max():.4g}")
            if vals.median() > 1.5:
                print("Note: SSM values look like percentage units. Consider converting to 0-1 volumetric fraction before scaling.")
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
