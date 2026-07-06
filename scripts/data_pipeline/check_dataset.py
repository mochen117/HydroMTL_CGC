"""
Ultimate Batch NetCDF Dataset Inspector.

Fuses file integrity checks (ds.load()), global attribute scanning, 
and unique categorical value extraction into a single formatted report.
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List
from tqdm import tqdm

def _process_data(stats: Dict[str, Any], data: Any, source_type: str) -> None:
    """Helper function to update statistics for a given array/scalar."""
    stats['source'].add(source_type)
    arr = np.asarray(data)
    
    # Handle string/object types (e.g., categorical text or global attributes like 'huc_02'="01")
    if arr.dtype.kind in ('S', 'U', 'O'):
        stats['is_numeric'] = False
        stats['dtypes'].add('string')
        arr_flat = arr.ravel()
        stats['total_count'] += arr_flat.size
        
        valid_vals = [v for v in arr_flat if v is not None and str(v).lower() not in ('nan', 'na', 'none')]
        stats['nan_count'] += (arr_flat.size - len(valid_vals))
        
        # Track unique values (cap at 35 to prevent memory issues with unique IDs)
        if len(stats['unique_vals']) < 35:
            for v in valid_vals:
                if isinstance(v, bytes):
                    v = v.decode('utf-8', errors='ignore')
                stats['unique_vals'].add(str(v))
        return

    # Handle numeric types (float/int)
    stats['dtypes'].add(str(arr.dtype))
    arr_flat = arr.ravel()
    nan_mask = np.isnan(arr_flat)
    nan_count = int(nan_mask.sum())
    valid_count = arr_flat.size - nan_count
    
    stats['total_count'] += arr_flat.size
    stats['nan_count'] += nan_count
    
    if valid_count > 0:
        valid_arr = arr_flat[~nan_mask]
        v_min = float(np.min(valid_arr))
        v_max = float(np.max(valid_arr))
        
        if v_min < stats['min_val']: stats['min_val'] = v_min
        if v_max > stats['max_val']: stats['max_val'] = v_max
        
        # Extract unique categories ONLY for integer types
        if np.issubdtype(arr.dtype, np.integer):
            if len(stats['unique_vals']) < 35:
                stats['unique_vals'].update(valid_arr.tolist())

def generate_ultimate_report(data_dir_path: str) -> None:
    data_dir = Path(data_dir_path)
    if not data_dir.is_dir():
        print(f"Error: Directory not found -> {data_dir}")
        return

    nc_files = list(data_dir.glob("*.nc"))
    if not nc_files:
        print(f"Error: No .nc files found in -> {data_dir}")
        return

    print(f"Starting rigorous inspection of {len(nc_files)} files...")

    var_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'source': set(),
        'dtypes': set(),
        'total_count': 0,
        'nan_count': 0,
        'min_val': float('inf'),
        'max_val': float('-inf'),
        'unique_vals': set(),
        'is_numeric': True
    })
    
    failed_files: List[tuple[str, str]] = []

    for file_path in tqdm(nc_files, desc="Scanning files", unit="file"):
        try:
            with xr.open_dataset(file_path) as ds:
                # STRICT CORRUPTION CHECK: Force load into memory
                ds.load() 
                
                # 1. Scan Data Variables
                for var_name, var_data in ds.variables.items():
                    _process_data(var_stats[var_name], var_data.values, 'Var')
                    
                # 2. Scan Global Attributes
                for attr_name, attr_val in ds.attrs.items():
                    _process_data(var_stats[attr_name], attr_val, 'Attr')
                    
        except Exception as e:
            failed_files.append((file_path.name, str(e)))

    # --- Print Report ---
    print("\n" + "=" * 105)
    print("ULTIMATE DATASET HEALTH REPORT (Attributes & Variables)")
    print("=" * 105)
    print(f"Total Files Scanned : {len(nc_files)}")
    print(f"Successfully Read   : {len(nc_files) - len(failed_files)}")
    print(f"Corrupted/Failed    : {len(failed_files)}")
    
    if failed_files:
        print("\n[!] Corrupted Files List:")
        for f, err in failed_files[:10]:
            print(f"  - {f}: {err}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more.")

    print("\n[Aggregated Statistics]")
    print("-" * 105)
    header = f"{'Name (Attr/Var)':<24} | {'Src':<5} | {'Dtypes':<10} | {'NaN %':<7} | {'Min/Max or Unique Categories'}"
    print(header)
    print("-" * 105)

    for var_name in sorted(var_stats.keys()):
        stats = var_stats[var_name]
        src_str = "/".join(sorted(list(stats['source'])))
        dtypes_str = ",".join(sorted(list(stats['dtypes'])))
        
        nan_pct = (stats['nan_count'] / stats['total_count']) * 100 if stats['total_count'] > 0 else 0
        nan_str = f"{nan_pct:>5.1f}%"

        # Display Logic: Show unique categories for strings/ints, else show Min->Max
        if stats['nan_count'] == stats['total_count'] and stats['total_count'] > 0:
            detail_str = "[All NaN / Missing]"
        elif not stats['is_numeric'] or 'int' in dtypes_str.lower():
            n_unique = len(stats['unique_vals'])
            if 0 < n_unique <= 30:
                # Sort numeric strings properly if possible
                try:
                    sorted_vals = sorted(list(stats['unique_vals']), key=float)
                except ValueError:
                    sorted_vals = sorted(list(stats['unique_vals']))
                detail_str = f"Categorical ({n_unique} unique): {sorted_vals}"
            elif n_unique > 30:
                detail_str = f"Categorical (>30 unique values, omitted)"
            else:
                detail_str = "No valid data"
        else:
            detail_str = f"{stats['min_val']:<8.4g} -> {stats['max_val']:<8.4g}"

        # Highlight 'huc' related fields
        prefix = ">> " if "huc" in var_name.lower() else "   "
        
        print(f"{prefix}{var_name:<21} | {src_str:<5} | {dtypes_str:<10} | {nan_str:<7} | {detail_str}")

    print("-" * 105)
    print("Inspection completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict batch inspect NetCDF files (variables & attributes).")
    parser.add_argument("directory", type=str, help="Directory containing .nc files.")
    args = parser.parse_args()
    
    generate_ultimate_report(args.directory)