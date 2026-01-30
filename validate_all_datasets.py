"""
Validate all Hydro-MTL datasets for compliance with paper requirements.
"""

import xarray as xr
import json
import pandas as pd
from pathlib import Path
import numpy as np


def validate_dataset(nc_file):
    """Validate a single dataset against paper requirements."""
    try:
        ds = xr.open_dataset(nc_file)
        gage_id = ds.attrs.get('gage_id', 'unknown')
        
        # Check requirements from the paper
        checks = {
            'time_period_correct': len(ds.time) == 7578,  # 2001-01-01 to 2021-09-30
            'streamflow_coverage_ge_95': ds.attrs.get('coverage_achieved', 0) >= 0.95,
            'all_variables_present': all(var in ds.data_vars for var in 
                                       ['streamflow', 'total_precipitation', 'temperature', 
                                        'et', 'pet', 'ssm', 'susm']),
            'cf_compliant': 'CF' in ds.attrs.get('Conventions', ''),
            'nldas2_source_correct': 'NLDAS-2' in ds.attrs.get('source', ''),
            'et_description_correct': 'resampled from 8-day to daily' in ds['et'].attrs.get('description', ''),
            'ssm_description_correct': '3-day timestep' in ds['ssm'].attrs.get('description', '')
        }
        
        # Variable-specific checks
        var_checks = {}
        for var_name in ['streamflow', 'total_precipitation', 'temperature', 'et', 'pet', 'ssm', 'susm']:
            if var_name in ds.data_vars:
                var_attrs = ds[var_name].attrs
                var_checks[f'{var_name}_has_units'] = 'units' in var_attrs
                var_checks[f'{var_name}_has_long_name'] = 'long_name' in var_attrs
                var_checks[f'{var_name}_has_standard_name'] = 'standard_name' in var_attrs
        
        ds.close()
        
        # Calculate validation score
        all_checks = {**checks, **var_checks}
        passed = sum(all_checks.values())
        total = len(all_checks)
        
        return {
            'gage_id': gage_id,
            'nc_file': nc_file.name,
            'coverage': ds.attrs.get('coverage_achieved', 0),
            'time_points': len(ds.time),
            'variables_present': list(ds.data_vars.keys()),
            'passed_checks': passed,
            'total_checks': total,
            'validation_score': passed / total if total > 0 else 0,
            'all_passed': passed == total,
            'specific_issues': [key for key, value in all_checks.items() if not value]
        }
        
    except Exception as e:
        return {
            'gage_id': 'unknown',
            'nc_file': nc_file.name,
            'coverage': 0,
            'time_points': 0,
            'variables_present': [],
            'passed_checks': 0,
            'total_checks': 0,
            'validation_score': 0,
            'all_passed': False,
            'specific_issues': [f'Error: {str(e)}']
        }


def validate_all_datasets():
    """Validate all generated NetCDF files."""
    output_dir = Path("/home/mochen/code/HydroMTL_CGC/output")
    nc_files = list(output_dir.glob("gage_*.nc"))
    
    print(f"=== Hydro-MTL Dataset Validation ===\n")
    print(f"Found {len(nc_files)} NetCDF files to validate.\n")
    
    results = []
    
    for i, nc_file in enumerate(nc_files):
        if i % 50 == 0:
            print(f"Validating file {i+1}/{len(nc_files)}...")
        
        result = validate_dataset(nc_file)
        results.append(result)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Generate summary statistics
    print(f"\n=== Validation Summary ===")
    print(f"Total files validated: {len(df)}")
    print(f"Files passing all checks: {df['all_passed'].sum()} ({df['all_passed'].mean():.1%})")
    print(f"Average validation score: {df['validation_score'].mean():.2%}")
    print(f"Average streamflow coverage: {df['coverage'].mean():.2%}")
    
    # Check source field specifically
    source_issues = []
    for nc_file in nc_files[:10]:  # Check first 10 files
        try:
            ds = xr.open_dataset(nc_file)
            source = ds.attrs.get('source', '')
            ds.close()
            if 'Daymet' in source:
                source_issues.append((ds.attrs.get('gage_id', 'unknown'), source))
        except:
            pass
    
    if source_issues:
        print(f"\n⚠️ Source field issues found (Daymet instead of NLDAS-2):")
        for gage_id, source in source_issues[:5]:
            print(f"  {gage_id}: {source}")
    
    # Save detailed report
    report_file = output_dir / "validation_report.csv"
    df.to_csv(report_file, index=False)
    print(f"\nDetailed validation report saved to: {report_file}")
    
    # Save summary report
    summary = {
        'validation_date': pd.Timestamp.now().isoformat(),
        'total_files': len(df),
        'files_passing_all': int(df['all_passed'].sum()),
        'pass_rate': float(df['all_passed'].mean()),
        'avg_validation_score': float(df['validation_score'].mean()),
        'avg_coverage': float(df['coverage'].mean()),
        'common_issues': df['specific_issues'].explode().value_counts().head(10).to_dict()
    }
    
    summary_file = output_dir / "validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Validation summary saved to: {summary_file}")
    
    return df


if __name__ == "__main__":
    validate_all_datasets()